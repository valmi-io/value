"""
Copyright (c) 2024 valmi.io <https://github.com/valmi-io>

Created Date: Tuesday, March 21st 2023, 6:56:21 pm
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
from datetime import datetime
from pydantic import UUID4
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session
from sqlalchemy import func, text,cast
from sqlalchemy import  Text
from sqlalchemy.dialects.postgresql import JSONB
from metastore.models import SyncRun
from api.schemas import SyncRunCreate
from typing import Any, Dict, List, Optional

from metastore.models import SyncStatus
from api.schemas.sync import LastSuccessfulSync, LatestSyncInfo
from .base import BaseService
from sqlalchemy.orm.attributes import flag_modified
import logging 
logger = logging.getLogger(__name__)


class SyncRunsService(BaseService[SyncRun, SyncRunCreate, Any]):
    def __init__(self, db_session: Session):
        super(SyncRunsService, self).__init__(SyncRun, db_session)

    def create(self, obj: SyncRunCreate) -> SyncRun:
        return super().create(obj)

    def get_runs(self, sync_id: UUID4, before: datetime, limit: int) -> List[SyncRun]:
        return (
            self.db_session.query(self.model)
            .filter(and_(SyncRun.run_at < before), SyncRun.sync_id == sync_id)
            .order_by(SyncRun.run_at.desc())
            .limit(limit)
            .all()
        )

    def get_run(self, sync_id: UUID4, run_id: UUID4) -> SyncRun:
        return (
            self.db_session.query(self.model)
            .filter(SyncRun.sync_id == sync_id, SyncRun.id == run_id)
            .first()
        )

    def get_active_or_latest_runs(self, after: datetime) -> List[SyncRun]:
        return (
            self.db_session.query(self.model)
            .filter(or_(SyncRun.run_at > after, SyncRun.status != SyncStatus.STOPPED.value))
            .all()
        )

    def save_status(self, sync_id, run_id, connector_string, status):
        sync_run = self.get(run_id)
        self.db_session.refresh(sync_run)
        if not sync_run.extra:
            sync_run.extra = {}
        if connector_string not in sync_run.extra:
            sync_run.extra[connector_string] = {}

        sync_run.extra[connector_string]["status"] = status
        flag_modified(sync_run, "extra")

        self.db_session.commit()

    def save_state(self, sync_id, run_id, connector_string, mode, state):
        sync_run = self.get(run_id)
        self.db_session.refresh(sync_run)

        if not sync_run.extra:
            sync_run.extra = {}
        if connector_string not in sync_run.extra:
            sync_run.extra[connector_string] = {}

        # PER STREAM STATE message in etl sources
        if mode == "etl" and connector_string == "src":
            state_to_input: List = None
            _valmi_meta = None
            if "_valmi_meta" in state:
                _valmi_meta = state['_valmi_meta']
                del state['_valmi_meta']

            if "state" in sync_run.extra[connector_string]:
                state_to_input = sync_run.extra[connector_string]["state"]

            if state_to_input is None:
                state_to_input = {}
                state_to_input["global"] = [state]
            else:
                current_stream: str = state['stream']['stream_descriptor']['name']
                new_state = []
                for s in state_to_input["global"]:
                    print(s)
                    if current_stream != s['stream']['stream_descriptor']['name']:
                        new_state.append(s)
                new_state.append(state)
                state_to_input["global"] = new_state
            state_to_input["_valmi_meta"] = _valmi_meta
        else:
            state_to_input = state

        sync_run.extra[connector_string]["state"] = state_to_input
        flag_modified(sync_run, "extra")

        self.db_session.commit()

    def update_sync_run_extra_data(self, run_id, connector_string, key, value):
        sync_run = self.get(run_id)
        self.db_session.refresh(sync_run)

        if not sync_run.extra:
            sync_run.extra = {}

        if connector_string not in sync_run.extra:
            sync_run.extra[connector_string] = {}

        sync_run.extra[connector_string][key] = value
        flag_modified(sync_run, "extra")

        self.db_session.commit()

    def last_successful_sync_run(self, sync_id) -> LastSuccessfulSync:
        try:
            logger.debug('here in activation below is query')

            # Query for the latest stopped sync run
            result = (
                self.db_session.query(self.model)
                .filter(
                    SyncRun.sync_id == sync_id,
                    func.json_extract_path_text(SyncRun.extra, 'src', 'status', 'status') == 'success',
                    func.json_extract_path_text(SyncRun.extra, 'dest', 'status', 'status') == 'success',
                    func.json_extract_path_text(SyncRun.extra, 'run_manager', 'status', 'status') == 'success',
                )
                .order_by(SyncRun.created_at.desc())
                .limit(1)
                .first()
            )
            logger.debug("query executed")
            if result is None:
                return LastSuccessfulSync(found=False)
            return LastSuccessfulSync(found=True,run_end_at = result.run_end_at)
        except Exception as e:
            logger.error(e)
    def latest_sync_info(self, sync_id)->LatestSyncInfo:
        try:
            result = (
                self.db_session.query(self.model)
                .filter(SyncRun.sync_id == sync_id)
                .order_by(SyncRun.created_at.desc())
                .limit(1)
                .first()
            )
            logger.debug(result)
            if result is None:
                return LatestSyncInfo(found=False)
            if result.extra is None:
                return LatestSyncInfo(found=True, status="running", created_at=result.created_at)
            src_status = result.extra.get("src", {}).get("status", {}).get("status")
            dest_status = result.extra.get("dest", {}).get("status", {}).get("status")
            run_manager_status = result.extra.get("run_manager", {}).get("status", {}).get("status")

            if src_status is None or dest_status is None or run_manager_status is None:
                status = "running"
            elif src_status == "success" and dest_status == "success" and run_manager_status == "success":
                status = "success"
            else:
                status = "failed"
            return LatestSyncInfo(found=True,status=status,created_at=result.created_at) 
        except Exception as e:
            logger.debug("in exception")
            logger.debug(e)
            logger.exception(e)