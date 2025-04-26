from datetime import timedelta

from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from snowflake.core import Root, CreateMode
from snowflake.core.database import Database
from snowflake.core.schema import Schema
from snowflake.core.stage import Stage
from snowflake.core.table import Table, TableColumn, PrimaryKey
from snowflake.core.task import StoredProcedureCall, Task
from snowflake.core.task.dagv1 import DAGOperation, DAG, DAGTask
from snowflake.core.warehouse import Warehouse