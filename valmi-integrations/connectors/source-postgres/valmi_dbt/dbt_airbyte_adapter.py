"""
Copyright (c) 2024 valmi.io <https://github.com/valmi-io>

Created Date: Tuesday, March 14th 2023, 1:49:58 pm
Author: Rajashekar Varkala @ valmi.io

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from jinja2 import Environment, FileSystemLoader

from fal import FalDbt
import dbt.adapters.factory as adapters_factory
import os
from dbt.adapters.sql import SQLAdapter
from dbt.adapters.base.relation import BaseRelation
from airbyte_cdk.logger import AirbyteLogger
from faldbt import lib, parse
from airbyte_protocol.models import SyncMode
from dbt.events.functions import cleanup_event_logger
import json
import subprocess
from airbyte_cdk.models import (
    ConfiguredAirbyteCatalog,
)
import glob
import shlex
from dbt.tracking import do_not_track
from faldbt.logger import LOGGER
import logging


class CustomFalDbt(FalDbt):
    def __init__(self, *args, **kwargs) -> None:
        """
        if "_basic" in kwargs and kwargs["_basic"] is False:
            del kwargs["_basic"]
            super(CustomFalDbt, self).__init__(*args, **kwargs)
            cleanup_event_logger()

        else:
        """

        # UGLIEST Hack
        LOGGER._logger = logging.getLogger("airbyte")

        self.project_dir = os.path.realpath(os.path.expanduser(kwargs["project_dir"]))
        self.profiles_dir = os.path.realpath(os.path.expanduser(kwargs["profiles_dir"]))

        lib.initialize_dbt_flags(profiles_dir=self.profiles_dir)

        self._config = parse.get_dbt_config(
            project_dir=self.project_dir,
            profiles_dir=self.profiles_dir,
        )

        lib.register_adapters(self._config)

        if "_basic" in kwargs and kwargs["_basic"] is False:
            self._run_results = parse.get_dbt_results(self.project_dir, self._config)

        cleanup_event_logger()


class DbtAirbyteAdpater:
    def get_fal_dbt(self, _basic=True):
        faldbt = CustomFalDbt(
            _basic=_basic,
            profiles_dir=self.get_abs_path("valmi_dbt_source_transform"),
            project_dir=self.get_abs_path("valmi_dbt_source_transform"),
        )
        do_not_track()
        return faldbt

    def get_jinja_template(self, logger: AirbyteLogger, template_fname):
        file_loader = FileSystemLoader(self.get_cur_dir())
        env = Environment(loader=file_loader)
        template = env.get_template(template_fname)
        return template

    def get_cur_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def get_abs_path(self, path):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    def write_profiles_config_from_spec(self, logger: AirbyteLogger, config):
        template = self.get_jinja_template(logger, "profiles_template.jinja")
        output = template.render(config=config)
        with open(self.get_abs_path("valmi_dbt_source_transform/profiles.yml"), "w") as f:
            f.write(output)

    def check_connection(self):
        faldbt = self.get_fal_dbt()
        adapter: SQLAdapter = adapters_factory.get_adapter(faldbt._config)
        adapter.connections.open(adapter.acquire_connection("check_connection"))

    def discover_streams(self, logger: AirbyteLogger, config):
        self.faldbt = self.get_fal_dbt()

        self.adapter: SQLAdapter = adapters_factory.get_adapter(self.faldbt._config)

        with self.adapter.connection_named("discover-connection"):
            if "namespace" in config:
                relations = self.adapter.list_relations(
                    self.faldbt._config.credentials.database, schema=config["namespace"])
                if "query" in config:
                    adapter_resp, agate_table = self.adapter.execute(config["query"], fetch=True)
                    # adapter_resp, agate_table = self.execute_sql(self.faldbt, logger, config["query"])
                    # logger.info(agate_table)
                    return (False, agate_table)
                if "table" in config:
                    # get subset of the relations
                    subset = []
                    for relation in relations:
                        if relation.path.identifier.lower() == config["table"].lower() \
                                and relation.path.schema.lower() == config["namespace"].lower():
                            subset.append(relation)
                            break
                    return (
                        False,
                        subset
                    )
                return (
                    False,
                    self.adapter.list_relations(self.faldbt._config.credentials.database, schema=config["namespace"]),
                )
            return (True, self.adapter.list_schemas(self.faldbt._config.credentials.database))

    def get_columns(self, adapter: SQLAdapter, relation: BaseRelation):
        with self.adapter.connection_named("getcolumns-connection"):
            return adapter.get_columns_in_relation(relation)

    def generate_dummy_project_yml(self, logger: AirbyteLogger):
        template = self.get_jinja_template(logger, "dbt_project.jinja")

        class Args(object):
            def __getitem__(self, arg):
                return ""

        args = Args()

        output = template.render(args=args)
        with open(self.get_abs_path("valmi_dbt_source_transform/dbt_project.yml"), "w") as f:
            f.write(output)

    def generate_project_yml(self, logger: AirbyteLogger, config: json, catalog: ConfiguredAirbyteCatalog, sync_id):
        template = self.get_jinja_template(logger, "dbt_project.jinja")

        full_refresh = "false"
        if catalog.streams[0].sync_mode == SyncMode.full_refresh:
            full_refresh = "true"

        # override full_refresh from run_time_args
        if "run_time_args" in config and "full_refresh" in config["run_time_args"]:
            full_refresh = config["run_time_args"]["full_refresh"]

        previous_run_status = "success"
        # override previous_run_status from run_time_args
        if "run_time_args" in config and "previous_run_status" in config["run_time_args"]:
            previous_run_status = config["run_time_args"]["previous_run_status"]

        col_arr_str = ",".join([f'"{col}"' for col in catalog.streams[0].stream.json_schema["properties"].keys()])
        args = {
            "sync_id": sync_id,
            "full_refresh": full_refresh,
            "columns": f"[{col_arr_str}]",
            "id_key": catalog.streams[0].id_key,
            "name": self.get_table_name(catalog.streams[0].stream.name),
            "query": config["query"],
            "previous_run_status": previous_run_status,
            "destination_sync_mode": catalog.streams[0].destination_sync_mode.value,
        }

        output = template.render(args=args)
        with open(self.get_abs_path("valmi_dbt_source_transform/dbt_project.yml"), "w") as f:
            f.write(output)

    def generate_source_yml(self, logger: AirbyteLogger, config, catalog, sync_id):
        template = self.get_jinja_template(logger, "source_schema.jinja")

        source = {
            "stream": self.get_table_name(catalog.streams[0].stream.name),
            "schema": self.get_namespace(catalog.streams[0].stream.name),
            "database": config["database"],
            "sync_id": sync_id,
        }

        output = template.render(source=source)
        with open(self.get_abs_path("valmi_dbt_source_transform/models/staging/schema.yml"), "w") as f:
            f.write(output)

    def execute_sql(self, faldbt, sql: str):
        compiled_result = lib.compile_sql(
            faldbt.project_dir,
            faldbt.profiles_dir,
            faldbt._profile_target,
            sql,
            config=faldbt._config,
        )
        # NOTE: changed in version 1.3.0 to `compiled_code`
        if hasattr(compiled_result, "compiled_code"):
            sql = compiled_result.compiled_code
        else:
            sql = compiled_result.compiled_sql

        adapter: SQLAdapter = adapters_factory.get_adapter(faldbt._config)  # type: ignore
        with adapter.connection_named("faldbt"):
            return adapter.execute(sql, fetch=True)

    def execute_dbt(self, logger: AirbyteLogger):
        logger.info("Initiating dbt run")

        cmd_arr = shlex.split(
            f"dbt --log-format json run --profiles-dir {self.get_abs_path('valmi_dbt_source_transform')} \
                --project-dir {self.get_abs_path('valmi_dbt_source_transform')}"
        )
        proc = subprocess.Popen(
            cmd_arr,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        logs = []
        for line in iter(proc.stdout.readline, b''):
            logs.append(line)
        proc.stdout.close()

        while True:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.info("Waiting for dbt to finish...")
                continue
            break

        lastError = None
        for log in logs:
            j = json.loads(log.decode())
            if j["info"]["level"].lower() == "error":
                lastError = j

        if proc.returncode is not None and proc.returncode != 0:
            raise Exception(lastError["info"]["msg"])
        else:
            logger.info("Dbt run successful")

    def append_sql_files_with_sync_id(self, sync_id: str):
        for filename in glob.iglob(self.get_abs_path("valmi_dbt_source_transform/**/*.sql"), recursive=True):
            nosuffix = filename[:-4]
            if not nosuffix.endswith(sync_id):
                os.rename(filename, nosuffix + "_" + sync_id + ".sql")

    def get_table_name(self, full_path):
        return full_path.split(".")[-1].strip('"')

    def get_namespace(self, full_path):
        return full_path.split(".")[-2].strip('"')
