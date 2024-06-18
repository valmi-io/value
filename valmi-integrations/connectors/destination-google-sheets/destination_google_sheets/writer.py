#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#


from airbyte_cdk.models import AirbyteStream
from pygsheets import Worksheet
from valmi_connector_lib.valmi_protocol import ValmiSink

from .buffer import WriteBufferMixin
from .spreadsheet import GoogleSheets


class GoogleSheetsWriter(WriteBufferMixin):
    def __init__(self, spreadsheet: GoogleSheets, stream_name: str):
        self.spreadsheet = spreadsheet
        self.stream_name = stream_name
        super().__init__(self.stream_name)

    def delete_stream_entries(self):
        """
        Deletes all the records belonging to the input stream.
        """
        self.spreadsheet.clean_worksheet(self.stream_name)

    def check_headers(self):
        """
        Checks whether data headers belonging to the input stream are set.
        """
        stream = self.stream_info[self.stream_name]
        if not stream["is_set"]:
            self.spreadsheet.set_headers(self.stream_name, stream["headers"])
            self.stream_info[self.stream_name]["is_set"] = True

    def queue_write_operation(self, stream: str):
        """
        Mimics `batch_write` operation using records_buffer.

        1) gets data from the records_buffer, with respect to the size of the records_buffer (records count or size in Kb)
        2) writes it to the target worksheet
        3) cleans-up the records_buffer belonging to input stream
        """
        # get the size of records_buffer for target stream in Kb
        # TODO unit test flush triggers
        records_buffer_size_in_kb = self.records_buffer[self.stream_name].__sizeof__() / 1024
        if (
            len(self.records_buffer[self.stream_name]) == self.flush_interval
            or records_buffer_size_in_kb > self.flush_interval_size_in_kb
        ):
            self.write_from_queue()
            self.clear_buffer()
            return True
        return False

    def write_from_queue(self):
        """
        Writes data from records_buffer for belonging to the input stream.

        1) checks the headers are set
        2) gets the values from the records_buffer
        3) if there are records to write - writes them to the target worksheet
        """

        self.check_headers()
        values: list = self.records_buffer[self.stream_name] or []
        if values:
            stream: Worksheet = self.spreadsheet.open_worksheet(self.stream_name)
            self.logger.info(f"Writing data for stream: {self.stream_name}")
            # we start from the cell of `A2` as starting range to fill the spreadsheet
            stream.append_table(values, start="A2", dimension="ROWS")
        else:
            self.logger.info(f"Skipping empty stream: {self.stream_name}")

    def write_whats_left(self):
        """
        Stands for writing records that are still left to be written,
        but don't match the condition for `queue_write_operation`.
        """
        for stream_name in self.records_buffer:
            self.write_from_queue()
            self.clear_buffer()

    def deduplicate_records(self, configured_stream: AirbyteStream, sink: ValmiSink):
        """
        Finds and removes duplicated records for target stream, using `primary_key`.
        Processing the worksheet happens offline and sync it afterwards to reduce API calls rate
        If rate limits are hit while deduplicating, it will be handeled automatically, the operation continues after backoff.
        """
        src_fields = list(map(lambda map_obj: (map_obj["stream"], map_obj,), sink.mapping))
        filtered_src_fields = list(filter(lambda x: x[0] == configured_stream.id_key, src_fields))

        primary_key: str = (
            filtered_src_fields[0][1]["sink"]
            if len(filtered_src_fields) > 0
            else configured_stream.id_key
        )

        stream: Worksheet = self.spreadsheet.open_worksheet(self.stream_name)
        rows_to_remove: list = self.spreadsheet.find_duplicates(stream, primary_key)

        if rows_to_remove:
            self.logger.info(f"Duplicated records are found for stream: {self.stream_name}, resolving...")
            self.spreadsheet.remove_duplicates(stream, rows_to_remove)
            self.logger.info(f"Finished deduplicating records for stream: {self.stream_name}")
        else:
            self.logger.info(f"No duplicated records found for stream: {self.stream_name}")
