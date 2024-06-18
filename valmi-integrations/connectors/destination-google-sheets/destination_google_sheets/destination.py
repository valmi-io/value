"""
Copyright (c) 2024 valmi.io <https://github.com/valmi-io>

Created Date: Wednesday, March 8th 2023, 11:38:42 am
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
from typing import Any, Dict, Iterable, Mapping

from airbyte_cdk import AirbyteLogger
from airbyte_cdk.destinations import Destination
from airbyte_cdk.models import AirbyteConnectionStatus, AirbyteMessage
from airbyte_cdk.models.airbyte_protocol import Status
from google.auth.exceptions import RefreshError
from valmi_connector_lib.destination_wrapper.destination_write_wrapper import (
    DestinationWriteWrapper, HandlerResponseData)
from valmi_connector_lib.valmi_destination import ValmiDestination
from valmi_connector_lib.valmi_protocol import (
    ConfiguredValmiCatalog, ConfiguredValmiDestinationCatalog,
    DestinationSyncMode, FieldCatalog, ValmiDestinationCatalog, ValmiSink)

from .client import GoogleSheetsClient
from .helpers import ConnectionTest, get_spreadsheet_id
from .spreadsheet import GoogleSheets
from .writer import GoogleSheetsWriter


class SheetsWriter(DestinationWriteWrapper):
    def modify_stream_name(self, stream_name: str) -> str:
        splits = stream_name.split(".")
        if len(splits) == 3:
            return splits[2]
        return stream_name
    def initialise_message_handling(self):
        spreadsheet_id = get_spreadsheet_id(self.config["spreadsheet_id"])

        client = GoogleSheetsClient(self.config).authorize()
        spreadsheet = GoogleSheets(client, spreadsheet_id)
        stream_name = self.modify_stream_name(self.configured_catalog.streams[0].stream.name)
        self.writer = GoogleSheetsWriter(spreadsheet, stream_name)

        self.writer.init_buffer_sink(
            configured_stream=self.configured_catalog.streams[0], sink=self.configured_destination_catalog.sinks[0]
        )

    def handle_message(
        self,
        msg,
        counter,
    ) -> HandlerResponseData:
        # _valmi_meta = msg.record.data['_valmi_meta']
        self.writer.add_to_buffer(msg.record.stream, msg.record.data)
        flushed = self.writer.queue_write_operation(msg.record.stream)

        # Google Sheets normalizes record by popping _valmi_meta . So setting it back to record.data
        # msg.record.data['_valmi_meta'] = _valmi_meta
        return HandlerResponseData(flushed=flushed)
    
    def finalise_message_handling(self):
        # if there are any records left in buffer
        self.writer.write_whats_left()
        # deduplicating records for `upsert` mode
        self.writer.deduplicate_records(self.configured_catalog.streams[0], self.configured_destination_catalog.sinks[0])


class DestinationGoogleSheets(ValmiDestination):
    def __init__(self):
        super().__init__()
        Destination.VALID_CMDS = {"spec", "check", "discover", "write"}

    def write(
        self,
        logger: AirbyteLogger,
        config: Mapping[str, Any],
        configured_catalog: ConfiguredValmiCatalog,
        configured_destination_catalog: ConfiguredValmiDestinationCatalog,
        input_messages: Iterable[AirbyteMessage],
        state: Dict[str, any],
    ) -> Iterable[AirbyteMessage]:
        sheets_writer = SheetsWriter(logger, config, configured_catalog, configured_destination_catalog, state)
        return sheets_writer.start_message_handling(input_messages)

    def discover(self, logger: AirbyteLogger, config: json) -> ValmiDestinationCatalog:
        sinks = []
        basic_field_catalog = FieldCatalog(
            json_schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
            },
            allow_freeform_fields=True,
            supported_destination_ids=[],
        )
        sinks.append(
            ValmiSink(
                name="Google Sheets",
                label="GoogleSheets",
                supported_destination_sync_modes=[
                    DestinationSyncMode.upsert
                ],
                field_catalog={
                    DestinationSyncMode.upsert.value: basic_field_catalog
                },
            )
        )
        return ValmiDestinationCatalog(sinks=sinks)

    def check(self, logger: AirbyteLogger, config: Mapping[str, Any]) -> AirbyteConnectionStatus:
        """
        Connection check method for Google Spreadsheets.
        Info:
            Checks whether target spreadsheet_id is available using provided credentials.
        Returns:
            :: Status.SUCCEEDED - if creadentials are valid, token is refreshed, target spreadsheet is available.
            :: Status.FAILED - if could not obtain new token, target spreadsheet is not available or other exception occured (with message).
        """
        spreadsheet_id = get_spreadsheet_id(config["spreadsheet_id"])
        try:
            client = GoogleSheetsClient(config).authorize()
            spreadsheet = GoogleSheets(client, spreadsheet_id)
            check_result = ConnectionTest(spreadsheet).perform_connection_test()
            if check_result:
                return AirbyteConnectionStatus(status=Status.SUCCEEDED)
        except RefreshError as token_err:
            return AirbyteConnectionStatus(status=Status.FAILED, message=f"{token_err}")
        except Exception as err:
            return AirbyteConnectionStatus(status=Status.FAILED, message=f"An exception occurred: {repr(err)}")

