#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#


from typing import Any, Mapping

from airbyte_cdk import AirbyteLogger
from airbyte_cdk.models import AirbyteStream
from valmi_connector_lib.valmi_protocol import ConfiguredValmiSink


class WriteBufferMixin:
    # Default instance of AirbyteLogger
    logger = AirbyteLogger()
    # intervals after which the records_buffer should be cleaned up for selected stream
    flush_interval = 500  # records count
    flush_interval_size_in_kb = 10 ^ 8  # memory allocation ~ 97656 Kb or 95 Mb

    def __init__(self, stream_name: str):
        # Buffer for input records
        self.records_buffer = {}
        # Placeholder for streams metadata
        self.stream_info = {}
        self.stream_name = stream_name

    @property
    def default_missing(self) -> str:
        """
        Default value for missing keys in record stream, compared to configured_stream catalog.
        Overwrite if needed.
        """
        return ""

    def init_buffer_sink(self, configured_stream: AirbyteStream, sink: ConfiguredValmiSink):
        """
        Saves important stream's information for later use.

        Particulary, creates the data structure for `records_stream`.
        Populates `stream_info` placeholder with stream metadata information.
        """
        stream = configured_stream.stream
        self.records_buffer[self.stream_name] = []

        # TODO: can be done much better

        src_fields = list(
            map(
                lambda map_obj: (
                    map_obj["stream"],
                    map_obj,
                ),
                sink.mapping,
            )
        )
        # dst_fields = list(map(lambda map_obj: (map_obj["sink"], map_obj,), sink.mapping))
        headers = []
        stream_props_used = []
        sorted_stream_keys = sorted(list(stream.json_schema.get("properties").keys()))
        for key in sorted_stream_keys:
            filtered_src_fields = list(filter(lambda x: x[0] == key, src_fields))
            if len(filtered_src_fields) > 0:
                for src_field_obj in filtered_src_fields:
                    headers.append(src_field_obj[1]["sink"])
                    stream_props_used.append(
                        (
                            src_field_obj[0],
                            src_field_obj[1]["sink"],
                        )
                    )
            else:
                headers.append(key)
                stream_props_used.append(
                    (
                        key,
                        key,
                    )
                )

        sorted_headers = sorted(headers)
        sorted_stream_props_used = sorted(stream_props_used, key=lambda x: x[1])

        self.stream_info[self.stream_name] = {
            "headers": sorted_headers,
            "stream_properties": list(map(lambda x: x[0], sorted_stream_props_used)),
            "is_set": False,
        }

    def add_to_buffer(self,stream_name:str, record: Mapping):
        """
        Populates input records to `records_buffer`.

        1) normalizes input record
        2) coerces normalized record to str
        3) gets values as list of record values from record mapping.
        """
        norm_record = self._normalize_record(self.stream_name, record)
        norm_values = list(map(str, norm_record.values()))
        self.records_buffer[self.stream_name].append(norm_values)

    def clear_buffer(self):
        """
        Cleans up the `records_buffer` values, belonging to input stream.
        """
        self.records_buffer[self.stream_name].clear()

    def _normalize_record(self, stream_name: str, record: Mapping) -> Mapping[str, Any]:
        """
        Updates the record keys up to the input configured_stream catalog keys.

        Handles two scenarios:
        1) when record has less keys than catalog declares (undersetting)
        2) when record has more keys than catalog declares (oversetting)

        Returns: alphabetically sorted, catalog-normalized Mapping[str, Any].

        EXAMPLE:
        - UnderSetting:
            * Catalog:
                - has 3 entities:
                    [ 'id', 'key1', 'key2' ]
                              ^
            * Input record:
                - missing 1 entity, compare to catalog
                    { 'id': 123,    'key2': 'value' }
                                  ^
            * Result:
                - 'key1' has been added to the record, because it was declared in catalog, to keep the data structure.
                    {'id': 123, 'key1': '', {'key2': 'value'} }
                                  ^
        - OverSetting:
            * Catalog:
                - has 3 entities:
                    [ 'id', 'key1', 'key2',   ]
                                            ^
            * Input record:
                - doesn't have entity 'key1'
                - has 1 more enitity, compare to catalog 'key3'
                    { 'id': 123,     ,'key2': 'value', 'key3': 'value' }
                                  ^                      ^
            * Result:
                - 'key1' was added, because it expected be the part of the record, to keep the data structure
                - 'key3' was dropped, because it was not declared in catalog, to keep the data structure
                    { 'id': 123, 'key1': '', 'key2': 'value',   }
                                   ^                          ^

        """
        data = {}

        stream_properties = self.stream_info[self.stream_name]["stream_properties"]
        headers = self.stream_info[self.stream_name]["headers"]

        for idx, prop in enumerate(stream_properties):
            data[headers[idx]] = record[prop]

        """
        These scenarios should not be there for valmi connector
        # undersetting scenario
        [record.update({headers[idx]: self.default_missing}) for idx, key in enumerate(stream_properties) if key not in record.keys()]
      
        self.logger.debug(str(record))

        # oversetting scenario
        [record.pop(key) for key in record.copy().keys() if key not in stream_properties]
        """
        # self.logger.debug(str(data))
        return dict(sorted(data.items(), key=lambda x: x[0]))
