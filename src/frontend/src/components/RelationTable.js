import React, { useEffect, useState } from "react";
import {
    Button,
    Col,
    Descriptions,
    Empty,
    Input,
    Row,
    Table,
    Tag,
    Tooltip,
} from "antd";
import { SearchOutlined } from "@ant-design/icons";
import Highlighter from "react-highlight-words";
import { generateUrl, mapLabelColor } from "./Utils";

const handleRelationProperties = (properties) => {
    let propertyContent = [];
    properties &&
        propertyContent.push(
            Object.keys(properties).forEach((k, idx) => {
                propertyContent.push(
                    <Descriptions.Item key={idx} label={k}>
                        {properties[k]}
                    </Descriptions.Item>
                );
            })
        );
    return (
        <Descriptions size="small" column={1} bordered={true}>
            {propertyContent}
        </Descriptions>
    );
};

const getPropertiesFilter = (relations) => {
    let propList = new Set();
    relations.forEach((entry) => {
        entry.properties &&
            Object.values(entry.properties).forEach((value) => {
                propList.add(value);
            });
    });
    return [...propList].map((val) => {
        return { text: val, value: val };
    });
};

const generateDataSource = (relations) => {
    return (
        relations &&
        relations.map((obj, idx) => {
            return {
                key: idx,
                relationName: obj.name,
                properties: obj.properties,
                target: obj.target,
            };
        })
    );
};

function RelationTable(props) {
    const [dataSource, setDataSource] = useState([]);
    const [searchInput, setSearchInput] = useState("");
    const [searchText, setSearchText] = useState("");
    const [searchedColumn, setSearchedColumn] = useState("");

    useEffect(() => {
        props.nodeRelations &&
            setDataSource(generateDataSource(props.nodeRelations));
    }, [props.nodeRelations]);

    const getColumnSearchProps = (dataIndex) => ({
        // enable search to filter table entries for a given search term
        filterDropdown: ({
            setSelectedKeys,
            selectedKeys,
            confirm,
            clearFilters,
        }) => (
            <div style={{ padding: 8 }}>
                <Input
                    ref={(node) => {
                        node && setSearchInput(node);
                    }}
                    placeholder={`Search ${dataIndex}`}
                    value={selectedKeys[0]}
                    onChange={(e) =>
                        setSelectedKeys(e.target.value ? [e.target.value] : [])
                    }
                    onPressEnter={() =>
                        handleSearch(selectedKeys, confirm, dataIndex)
                    }
                    style={{ width: "100%", marginBottom: 8, display: "block" }}
                />
                <Row justify="center">
                    <Col span={4}>
                        <Button
                            onClick={() => {
                                clearFilters();
                                handleSearch("", confirm, "");
                            }}
                            type="link"
                            size="small"
                            shape="default"
                            style={{ width: "100%" }}
                        >
                            Reset
                        </Button>
                    </Col>
                    <Col span={4} offset={16}>
                        <Button
                            onClick={() =>
                                handleSearch(selectedKeys, confirm, dataIndex)
                            }
                            type="primary"
                            size="small"
                            shape="default"
                            style={{ width: "100%" }}
                        >
                            Ok
                        </Button>
                    </Col>
                </Row>
            </div>
        ),
        filterIcon: (filtered) => (
            <SearchOutlined
                style={{ color: filtered ? "#1890ff" : undefined }}
            />
        ),
        onFilter: (value, record) => {
            if (dataIndex === "target") {
                return record.target.name
                    ? record.target.name
                          .toString()
                          .toLowerCase()
                          .includes(value.toLowerCase())
                    : "";
            } else if (dataIndex === "period") {
                if (Number.isInteger(parseInt(value))) {
                    return Array.isArray(record.target.period) &&
                        record.target.period.length === 2
                        ? parseInt(value) >= record.target.period[0] &&
                              parseInt(value) <= record.target.period[1]
                        : "";
                } else {
                    return record.target.period
                        ? record.target.period
                              .toString()
                              .toLowerCase()
                              .includes(value.toLowerCase())
                        : "";
                }
            } else {
                return record[dataIndex]
                    ? record[dataIndex]
                          .toString()
                          .toLowerCase()
                          .includes(value.toLowerCase())
                    : "";
            }
        },
        onFilterDropdownVisibleChange: (visible) => {
            if (visible) {
                setTimeout(() => searchInput, 100);
            }
        },
        render: (text) =>
            searchedColumn === dataIndex ? (
                <Highlighter
                    highlightStyle={{ backgroundColor: "#ffc069", padding: 0 }}
                    searchWords={[searchText]}
                    autoEscape
                    textToHighlight={text ? text.toString() : ""}
                />
            ) : (
                text
            ),
    });

    const handleSearch = (selectedKeys, confirm, dataIndex) => {
        confirm();
        setSearchText(selectedKeys[0] ? selectedKeys[0] : "");
        setSearchedColumn(dataIndex);
    };

    const columns = [
        {
            title: "Target",
            dataIndex: ["target", "name"],
            key: "target",
            ...getColumnSearchProps("target"),
            render: (entry) => (
                <Tooltip
                    placement="topLeft"
                    title="Click on the name to select the entry and examine it in more detail."
                >
                    {entry.toString().slice(0, -3)}
                </Tooltip>
            ),
            sorter: (a, b) =>
                a.target.name
                    .toString()
                    .localeCompare(b.target.name.toString()),
            onCell: (record) => {
                return {
                    onClick: () => {
                        props.setSelectedUri(record.target.uri);
                        //props.setSelectedLabel(record.name.replace("has", ""))
                    },
                };
            },
        },
        {
            title: "Period",
            dataIndex: ["target", "period"],
            key: "period",
            ...getColumnSearchProps("period"),
            render: (entry) => {
                if (Array.isArray(entry)) {
                    return entry.join(" - ");
                } else {
                    return entry;
                }
            },
            sorter: (a, b) =>
                a.target.period
                    .toString()
                    .localeCompare(b.target.period.toString()),
        },
        {
            title: "Locations",
            dataIndex: ["target", "locations"],
            key: "locations",
            render: (entry) => {
                if (Array.isArray(entry)) {
                    return entry.map((loc, idx) => {
                        return Array.isArray(loc) ? (
                            <Tag key={idx} color="processing">
                                {generateUrl(
                                    "https://www.openstreetmap.org/directions?from=" +
                                        loc[0] +
                                        "%2C" +
                                        loc[1] +
                                        "&to#map=5/" +
                                        loc[0] +
                                        "/" +
                                        loc[1] +
                                        "&layers=N",
                                    "(" + loc[0] + ", " + loc[1] + ")",
                                    idx
                                )}
                            </Tag>
                        ) : (
                            loc.toString()
                        );
                    });
                } else {
                    return entry;
                }
            },
        },
        {
            title: "Relation name",
            dataIndex: "relationName",
            key: "relationName",
            filters: dataSource
                ? [
                      ...new Set(dataSource.map((entry) => entry.relationName)),
                  ].map((value) => {
                      return {
                          text: value,
                          value: value,
                      };
                  })
                : [],
            filterMode: "tree",
            filterSearch: true,
            onFilter: (value, record) =>
                record.relationName
                    ? record.relationName
                          .toLowerCase()
                          .includes(value.toLowerCase())
                    : false,
            render: (entry, record) => {
                return (
                    <Tag color={mapLabelColor(record.target.label.toString())}>
                        {entry}
                    </Tag>
                );
            },
            sorter: (a, b) => a.relationName.localeCompare(b.relationName),
        },
        {
            title: "Properties",
            dataIndex: "properties",
            key: "properties",
            filters: props.nodeRelations
                ? getPropertiesFilter(props.nodeRelations)
                : [],
            filterMode: "tree",
            filterSearch: true,
            onFilter: (value, record) =>
                record.properties
                    ? Object.values(record.properties).includes(value)
                    : false,
            render: (entry) => {
                return handleRelationProperties(entry);
            },
        },
    ];

    return dataSource ? (
        <Table dataSource={dataSource} columns={columns} />
    ) : (
        <Empty />
    );
}

export default RelationTable;
