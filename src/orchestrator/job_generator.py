"""
Copyright (c) 2024 valmi.io <https://github.com/valmi-io>

Created Date: Thursday, March 9th 2023, 7:42:26 pm
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

import json
import logging
import os
import shutil
import threading
import time
from os.path import dirname, join
import requests
from requests.auth import HTTPBasicAuth
from dagster_graphql import ShutdownRepositoryLocationInfo, ShutdownRepositoryLocationStatus
from jinja2 import Environment, FileSystemLoader
from pydantic import Json
from vyper import v
from utils.retry_decorators import retry_on_exception, exception_to_sys_exit
from api.schemas import SyncScheduleCreate
from api.services import get_syncs_service
from metastore.session import get_session
from .dagster_client import ValmiDagsterClient
from api.services import SyncsService, SyncRunsService
from .run_manager import SyncRunnerThread

logger = logging.getLogger(v.get("LOGGER_NAME"))
GENERATED_DIR = "generated"
GENERATED_CONFIG_DIR = "config"
GENERATED_CATALOG_DIR = "catalog"
SHARED_DIR = "/tmp/shared_dir"

# TODO: clean it up with a better location
repo_ready = False


class JobCreatorThread(threading.Thread):
    def __init__(self, thread_id: int, name: str, dagster_client: ValmiDagsterClient,
                 sync_service: SyncsService, run_service: SyncRunsService) -> None:
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.exit_flag = False
        self.name = name
        self.dagster_client = dagster_client
        self.sync_service = sync_service
        self.run_service = run_service

    def run(self) -> None:
        global repo_ready
        while not self.exit_flag:
            try:
                resp = requests.get(
                    f"http://{v.get('APP_BACKEND')}:{v.get('APP_BACKEND_PORT')}/api/v1/superuser/syncs/",
                    timeout=v.get("HTTP_REQ_TIMEOUT"),
                    auth=HTTPBasicAuth(v.get("ADMIN_EMAIL"), v.get("ADMIN_PASSWORD")),
                )

                # TODO: improve strategy to detect new syncs
                syncs_changed: bool = False
                try:
                    with open(join(SHARED_DIR, f"{v.get('APP')}-syncs.json"), "r") as f:
                        if (f.read()) != resp.text:
                            syncs_changed = True
                except FileNotFoundError:
                    logging.exception("syncs.json not found")
                    syncs_changed = True

                # if new syncs are found, generate dagster jobs and restart dagster repository
                if syncs_changed:
                    logger.info("New syncs found")

                    # create directories for jobs, config and catalog
                    appdir = join(SHARED_DIR, v.get("APP"))
                    if not os.path.exists(appdir):
                        os.makedirs(appdir)

                    dirs = {
                        GENERATED_DIR: join(SHARED_DIR, v.get("APP"), "gen", GENERATED_DIR),
                        GENERATED_CONFIG_DIR: join(SHARED_DIR, v.get("APP"), "gen", GENERATED_CONFIG_DIR),
                        GENERATED_CATALOG_DIR: join(SHARED_DIR, v.get("APP"), "gen", GENERATED_CATALOG_DIR),
                    }
                    for key in dirs.keys():
                        dir = dirs[key]
                        if os.path.exists(dir):
                            shutil.rmtree(dir)
                        os.makedirs(dir)

                    syncs_json = resp.json()

                    # Insert into SYNC schedules metastore
                    self.insert_syncs_into_metastore(syncs_json)

                    # generate dagster jobs
                    self.gen_dagster_job_archive(dirs, syncs_json)

                    # restart dagster repository
                    self.restart_dagster_repo()

                    repo_ready = True

                    with open(join(SHARED_DIR, f'{v.get("APP")}-syncs.json'), "w") as replace_file:
                        replace_file.write(resp.text)
                else:
                    repo_ready = True

            except Exception:
                logger.exception("Error while fetching sync jobs and creating dagster jobs")
            time.sleep(1)

    def insert_syncs_into_metastore(self, syncs: Json[any]):

        sync_schedules = {}
        for sync in syncs:
            obj = SyncScheduleCreate(
                sync_id=sync["id"],
                status=sync["status"],
                run_interval=sync["schedule"].get("run_interval", 60000),
                src_connector_type=sync["source"]["credential"]["connector_type"],
                dst_connector_type=sync["destination"]["credential"]["connector_type"],
            )
            sync_schedules[obj.sync_id] = obj

        self.sync_service.insert_or_update_list_of_schedules(sync_schedules)
        # Not sure why the dbsession is giving stale values in run_manager - Check this later with active/disable sync run
        SyncRunnerThread.refresh_db_session()

    @exception_to_sys_exit
    @retry_on_exception
    def restart_dagster_repo(self) -> None:
        shutdown_info: ShutdownRepositoryLocationInfo = self.dagster_client.shutdown_repository_location(
            v.get("REPO_NAME")
        )
        if shutdown_info.status == ShutdownRepositoryLocationStatus.SUCCESS:
            logger.info("Dagster repo successfully shutdown")
        else:
            raise Exception(f"Repository location shutdown failed: {shutdown_info.message}")

    # make a zip archive of all dagster jobs
    def gen_dagster_job_archive(self, dirs: dict[str, str], syncs: Json[any]) -> None:
        print(syncs)
        for sync in syncs:
            self.gen_job_file(dirs, sync)
        # with open(join(dirs[GENERATED_DIR], f"__init__.py"), "w") as f:
        #    f.write("# Autogenerated: This file is needed to make a valid python package")
        shutil.make_archive(
            join(SHARED_DIR, f"{v.get('APP')}-valmi-jobs"), "zip", join(SHARED_DIR, v.get("APP"), "gen")
        )

    # use jinja2 to generate dagster a single job
    def gen_job_file(self, dirs: dict[str, str], sync: Json[any]) -> None:
        with open(join(dirs[GENERATED_CONFIG_DIR], f"{sync['id']}-{sync['source']['id']}.json"), "w") as f:
            f.write(json.dumps(sync["source"]["credential"]["connector_config"]))

        with open(join(dirs[GENERATED_CATALOG_DIR], f"{sync['id']}-{sync['source']['id']}.json"), "w") as f:
            f.write(json.dumps(sync["source"]["catalog"]))

        with open(join(dirs[GENERATED_CONFIG_DIR], f"{sync['id']}-{sync['destination']['id']}.json"), "w") as f:
            f.write(json.dumps(sync["destination"]["credential"]["connector_config"]))

        with open(join(dirs[GENERATED_CATALOG_DIR], f"{sync['id']}-{sync['destination']['id']}.json"), "w") as f:
            f.write(json.dumps(sync["destination"]["catalog"]))

        file_loader = FileSystemLoader(join(dirname(__file__), "templates"))
        env = Environment(loader=file_loader)
        template_name = "job_template.jinja"
        if sync["source"]["credential"]["connector_type"] == "SRC_SHOPIFY":
            template_name = "shopify_template.jinja"
        template = env.get_template(template_name)
        output = template.render(sync=sync, app=v.get("APP"), prefix=SHARED_DIR)
        with open(join(dirs[GENERATED_DIR], f"{sync['id'].replace('-','_')}.py"), "w") as f:
            f.write(output)
