from __future__ import annotations

import sys

import click

from dbtopo.config import ALL_DEPARTMENTS
from dbtopo.dedup import dedup_tables
from dbtopo.downloader import download_department
from dbtopo.extractor import extract_gpkg
from dbtopo.gpkg_reader import batch_ranges, layer_crs_epsg, list_layers
from dbtopo.schema import spark_schema_from_gpkg
from dbtopo.task_values import set_task_value
from dbtopo.writer import (
    _ingestion_schema,
    delete_department_rows,
    ensure_table_with_metadata,
    full_table_name,
)


def _parse_departments(departments: str) -> list[str]:
    if departments.strip().lower() == "all":
        return ALL_DEPARTMENTS
    return [d.strip() for d in departments.split(",")]


@click.group()
def main():
    """dbtopo - Load IGN BD TOPO into Databricks Delta tables."""
    pass


@main.command()
@click.option(
    "--departments", required=True, help="Comma-separated dept codes or 'all'"
)
@click.option("--catalog", required=True)
@click.option("--schema", default="ign_bdtopo")
@click.option("--volume", default="bronze_volume")
@click.option("--version", default="3-5")
@click.option("--projection", default="LAMB93")
@click.option("--version-date", default="2025-09-15")
@click.option("--skip-md5", is_flag=True, default=False, help="Skip MD5 verification")
def download_cmd(
    departments, catalog, schema, volume, version, projection, version_date, skip_md5
):
    """Download BD TOPO .7z archives to a Databricks Volume."""
    dept_list = _parse_departments(departments)
    volume_path = f"/Volumes/{catalog}/{schema}/{volume}"

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    for dept in dept_list:
        print(f"Downloading department {dept}...")
        path = download_department(
            dept=dept,
            volume_path=volume_path,
            version=version,
            projection=projection,
            version_date=version_date,
            verify_md5=not skip_md5,
        )
        set_task_value(spark, f"archive_path_{dept}", str(path))

    set_task_value(spark, "departments", dept_list)
    set_task_value(spark, "schema", schema)
    set_task_value(spark, "version", version)
    set_task_value(spark, "version_date", version_date)
    print(f"Download complete: {len(dept_list)} department(s).")


@main.command()
@click.option(
    "--departments", required=True, help="Comma-separated dept codes or 'all'"
)
@click.option("--catalog", required=True)
@click.option("--schema", default="ign_bdtopo")
@click.option("--volume", default="bronze_volume")
@click.option("--version", default="3-5")
@click.option("--projection", default="LAMB93")
@click.option("--version-date", default="2025-09-15")
@click.option(
    "--layers", default="", help="Comma-separated layer names, or empty for all"
)
@click.option(
    "--batch-size",
    default=5000,
    type=int,
    help="Rows per GPKG read. Each batch becomes one Spark task; "
    "keep under ~5K for serverless 1GB executor memory limit.",
)
@click.option("--table-prefix", default="ign_bdtopo_")
@click.option(
    "--lang",
    default="en",
    type=click.Choice(["en", "fr"]),
    help="Language for table/column metadata comments (default: en)",
)
def load_cmd(
    departments,
    catalog,
    schema,
    volume,
    version,
    projection,
    version_date,
    layers,
    batch_size,
    table_prefix,
    lang,
):
    """Extract GPKG from archives, transform, and load into Delta tables."""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    dept_list = _parse_departments(departments)
    volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
    layer_filter = [x.strip() for x in layers.split(",") if x.strip()] if layers else []

    rows_loaded: dict[str, int] = {}

    for dept in dept_list:
        dept_code = dept if dept.startswith("D") else f"D{dept}"
        base_name = (
            f"BDTOPO_{version}_TOUSTHEMES_GPKG_{projection}_{dept_code}_{version_date}"
        )
        archive_path = f"{volume_path}/{base_name}.7z"

        # Extract to Volume (not /tmp/) so Spark executors can access via FUSE.
        extract_dir = f"{volume_path}/extracted"
        print(f"Extracting {archive_path}...")
        gpkg_path = extract_gpkg(archive_path, output_dir=extract_dir)

        available_layers = list_layers(gpkg_path)
        target_layers = layer_filter if layer_filter else available_layers

        for layer_name in target_layers:
            if layer_name not in available_layers:
                print(f"  Skipping unknown layer: {layer_name}")
                continue

            table = full_table_name(catalog, schema, table_prefix, layer_name)
            print(f"  Loading layer {layer_name} -> {table}")

            from pyspark.sql.types import StringType

            layer_schema = spark_schema_from_gpkg(
                gpkg_path,
                layer_name,
                extra_columns={"dept": StringType(), "layer": StringType()},
            )

            # Pre-create table with metadata (column/table comments) before loading.
            ensure_table_with_metadata(
                spark,
                table,
                schema=layer_schema,
                layer=layer_name,
                version=version,
                version_date=version_date,
                lang=lang,
            )

            delete_department_rows(spark, table, dept_code)

            source_srid = layer_crs_epsg(gpkg_path, layer_name)
            total, ranges = batch_ranges(gpkg_path, layer_name, batch_size)

            if total == 0:
                print(f"    {layer_name}: 0 rows (empty layer)")
                rows_loaded[layer_name] = rows_loaded.get(layer_name, 0)
                set_task_value(spark, f"rows_{dept}_{layer_name}", 0)
                continue

            # Build ingestion schema: dates/timestamps as strings to avoid
            # pandas nanosecond overflow on historical dates (e.g. 1612-01-01).
            ingest_schema, cast_exprs = _ingestion_schema(layer_schema)
            date_ts_cols = list(cast_exprs.keys())

            # Create a Spark DF of batch ranges — one row per batch.
            gpkg_str = str(gpkg_path)
            range_rows = [
                (gpkg_str, layer_name, dept_code, offset, size)
                for _, offset, size in ranges
            ]
            range_df = spark.createDataFrame(
                range_rows,
                schema="gpkg_path string, layer string, dept string, "
                "offset int, batch_size int",
            )
            # One partition per batch → one Spark task per GPKG read.
            range_df = range_df.repartition(len(ranges))

            # UDF: each Spark task reads its GPKG slice, transforms, yields pandas DF.
            # NOTE: skip_features is O(1) on GPKG without Arrow mode
            # (pyogrio default). Adding use_arrow=True would regress.
            def _read_and_transform(iterator, _dt_cols=date_ts_cols):
                import geopandas as gpd_inner  # noqa: I001
                from dbtopo.transformer import transform_batch

                for pdf in iterator:
                    for _, row in pdf.iterrows():
                        gdf = gpd_inner.read_file(
                            row["gpkg_path"],
                            layer=row["layer"],
                            engine="pyogrio",
                            skip_features=int(row["offset"]),
                            max_features=int(row["batch_size"]),
                        )
                        if len(gdf) == 0:
                            continue
                        gdf, _ = transform_batch(
                            gdf, dept=row["dept"], layer=row["layer"]
                        )
                        # Convert date/timestamp cols to ISO strings so Arrow
                        # serialisation doesn't hit pandas ns-Timestamp limits.
                        import numpy as np

                        for c in _dt_cols:
                            if c in gdf.columns:
                                mask = gdf[c].isna()
                                gdf[c] = gdf[c].astype(str)
                                gdf.loc[mask, c] = np.nan
                        yield gdf

            result_df = range_df.mapInPandas(_read_and_transform, schema=ingest_schema)

            # Server-side: geometry conversion + date/timestamp casts.
            select_exprs: list[str] = []
            for field in layer_schema.fields:
                if field.name == "geometry":
                    geom_expr = f"ST_GeomFromWKT(geometry, {source_srid})"
                    if source_srid != 0 and source_srid != 4326:
                        geom_expr = f"ST_Transform({geom_expr}, 4326)"
                    select_exprs.append(f"{geom_expr} AS geometry")
                elif field.name in cast_exprs:
                    select_exprs.append(cast_exprs[field.name])
                else:
                    select_exprs.append(f"`{field.name}`")

            result_df = result_df.selectExpr(*select_exprs)

            # Write all batches to Delta in a single distributed write.
            try:
                result_df.write.format("delta").mode("append").option(
                    "mergeSchema", "true"
                ).saveAsTable(table)
            except Exception as exc:
                msg = str(exc)
                if "MEMORY_LIMIT" in msg:
                    raise RuntimeError(
                        f"Executor OOM writing {layer_name} for {dept_code} "
                        f"with batch_size={batch_size}. Each batch is read "
                        f"into a single executor (1 GB on serverless). "
                        f"Reduce --batch-size (current: {batch_size}) to "
                        f"lower per-task memory. Layers with complex "
                        f"geometries (e.g. batiment) need smaller batches."
                    ) from exc
                raise

            layer_rows = total
            rows_loaded[layer_name] = rows_loaded.get(layer_name, 0) + layer_rows
            set_task_value(spark, f"rows_{dept}_{layer_name}", layer_rows)
            print(f"    {layer_name}: {layer_rows} rows loaded ({len(ranges)} batches)")

    set_task_value(spark, "rows_total", rows_loaded)
    set_task_value(spark, "schema", schema)
    set_task_value(spark, "version", version)
    set_task_value(spark, "version_date", version_date)
    print("Load complete.")


@main.command()
@click.option("--catalog", required=True)
@click.option("--schema", default="ign_bdtopo")
@click.option("--table-prefix", default="ign_bdtopo_")
@click.option("--dedup-key", default="cleabs")
@click.option("--dedup-suffix", default="_dedup")
def dedup_cmd(catalog, schema, table_prefix, dedup_key, dedup_suffix):
    """Deduplicate all loaded tables by cleabs into *_dedup tables."""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    stats = dedup_tables(
        spark,
        catalog=catalog,
        schema=schema,
        table_prefix=table_prefix,
        dedup_key=dedup_key,
        dedup_suffix=dedup_suffix,
    )

    if not stats:
        print(f"No tables found with prefix '{table_prefix}' in {catalog}.{schema}")
        sys.exit(1)

    set_task_value(spark, "dedup_stats", stats)
    print("Deduplication complete.")


@main.command()
@click.option("--catalog", required=True)
@click.option("--schema", default="ign_bdtopo")
@click.option("--table-prefix", default="ign_bdtopo_")
def validate_cmd(catalog, schema, table_prefix):
    """Validate loaded data: row counts, GEOMETRY(4326), SRID, coordinates."""
    from pyspark.sql import SparkSession

    from dbtopo.validator import validate_tables

    spark = SparkSession.builder.getOrCreate()
    failures = validate_tables(spark, catalog, schema, table_prefix)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED: all checks succeeded.")


# Databricks python_wheel_task entry points.
# These call click commands with standalone_mode=False to avoid sys.exit(0).
def download():
    download_cmd(standalone_mode=False)


def load():
    load_cmd(standalone_mode=False)


def dedup():
    dedup_cmd(standalone_mode=False)


def validate():
    validate_cmd(standalone_mode=False)


if __name__ == "__main__":
    main()
