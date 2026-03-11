from __future__ import annotations

import logging

import pandas as pd
from pyspark.sql.types import (
    DateType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from dbtopo.metadata import get_column_descriptions, get_table_description

logger = logging.getLogger(__name__)


def delete_department_rows(spark, table_name: str, dept: str) -> None:
    """Delete existing rows for a department. No-op if table doesn't exist."""
    try:
        spark.sql(f"DELETE FROM {table_name} WHERE dept = '{dept}'")
    except Exception:
        pass  # Table doesn't exist yet on first run


def _escape_comment(text: str) -> str:
    """Escape single quotes for use in SQL COMMENT strings."""
    return text.replace("'", "\\'")


def ensure_table_with_metadata(
    spark,
    table_name: str,
    schema: StructType,
    layer: str,
    version: str = "3-5",
    version_date: str = "",
    crs: str = "EPSG:4326",
    lang: str = "en",
) -> None:
    """Create the table with column and table comments if it doesn't exist.

    Uses CREATE TABLE IF NOT EXISTS ... USING DELTA with per-column COMMENT
    clauses and a table-level COMMENT, so metadata is set at creation time.
    Also sets table properties for CRS and version info.

    Parameters
    ----------
    lang : str
        Language for metadata comments ("en" or "fr"). Defaults to "en".
    """
    col_descs = get_column_descriptions(layer, lang=lang)
    table_desc = get_table_description(
        layer, version=version, version_date=version_date, lang=lang
    )

    # Build column definitions from the Spark schema.
    # The geometry column is STRING in the StructType (WKT from pyogrio),
    # but must be declared as native GEOMETRY(srid) in the table so that
    # write_batch_to_delta's ST_GeomFromWKT output is schema-compatible
    # and the entire table is constrained to a single SRID.
    srid = crs.split(":")[1] if ":" in crs else "4326"
    col_defs = []
    for field in schema.fields:
        if field.name == "geometry":
            col_type = f"GEOMETRY({srid})"
        else:
            col_type = field.dataType.simpleString()
        col_sql = f"`{field.name}` {col_type}"
        if field.name in col_descs:
            col_sql += f" COMMENT '{_escape_comment(col_descs[field.name])}'"
        else:
            logger.warning(
                "No metadata for column '%s' in layer '%s'",
                field.name,
                layer,
            )
        col_defs.append(col_sql)

    cols_str = ",\n  ".join(col_defs)
    # Build TBLPROPERTIES inline to avoid a separate ALTER TABLE that races
    # under concurrent for_each_task iterations.
    props_parts = [f"'crs' = '{crs}'"]
    if version:
        props_parts.append(f"'bdtopo_version' = '{version}'")
    if version_date:
        props_parts.append(f"'bdtopo_version_date' = '{version_date}'")
    props_sql = ", ".join(props_parts)

    create_sql = (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        f"  {cols_str}\n"
        f") USING DELTA\n"
        f"COMMENT '{_escape_comment(table_desc)}'\n"
        f"TBLPROPERTIES ({props_sql})"
    )
    spark.sql(create_sql)


def _ingestion_schema(schema: StructType) -> tuple[StructType, dict[str, str]]:
    """Build a schema for createDataFrame with strings for date/timestamp cols.

    Returns (ingestion_schema, cast_exprs) where cast_exprs maps column names
    to SQL CAST expressions for server-side conversion.  This avoids Python-
    level row-by-row date parsing and sidesteps pandas' nanosecond Timestamp
    limits (which reject historical dates like "1612-01-01").
    """
    fields: list[StructField] = []
    casts: dict[str, str] = {}
    for field in schema.fields:
        if isinstance(field.dataType, DateType):
            fields.append(StructField(field.name, StringType(), nullable=True))
            casts[field.name] = f"TRY_CAST(`{field.name}` AS DATE) AS `{field.name}`"
        elif isinstance(field.dataType, TimestampType):
            fields.append(StructField(field.name, StringType(), nullable=True))
            casts[field.name] = (
                f"TRY_CAST(`{field.name}` AS TIMESTAMP) AS `{field.name}`"
            )
        else:
            fields.append(field)
    return StructType(fields), casts


def write_batch_to_delta(
    spark,
    pdf: pd.DataFrame,
    table_name: str,
    schema: StructType | None = None,
    source_srid: int = 0,
    target_srid: int = 4326,
) -> None:
    # Build a string-based ingestion schema so Arrow doesn't choke on dates,
    # then cast date/timestamp columns and convert geometry server-side in
    # a single selectExpr pass.
    cast_exprs: dict[str, str] = {}
    if schema is not None:
        ingest_schema, cast_exprs = _ingestion_schema(schema)
        sdf = spark.createDataFrame(pdf, schema=ingest_schema)
    else:
        sdf = spark.createDataFrame(pdf)

    # Build selectExpr: cast dates/timestamps + convert geometry, all at once.
    select_exprs: list[str] = []
    for col in sdf.columns:
        if col == "geometry":
            geom_expr = f"ST_GeomFromWKT(geometry, {source_srid})"
            if source_srid != 0 and source_srid != target_srid:
                geom_expr = f"ST_Transform({geom_expr}, {target_srid})"
            select_exprs.append(f"{geom_expr} AS geometry")
        elif col in cast_exprs:
            select_exprs.append(cast_exprs[col])
        else:
            select_exprs.append(f"`{col}`")

    if cast_exprs or "geometry" in sdf.columns:
        sdf = sdf.selectExpr(*select_exprs)

    sdf.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
        table_name
    )


def full_table_name(catalog: str, schema: str, table_prefix: str, layer: str) -> str:
    return f"{catalog}.{schema}.{table_prefix}{layer}"
