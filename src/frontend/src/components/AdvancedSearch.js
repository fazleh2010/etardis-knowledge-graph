import React, { useState } from "react";
import {
    Badge,
    Button,
    Col,
    Divider,
    Form,
    Input,
    Row,
    Select,
    Table,
    Tag,
    Tooltip,
} from "antd";
import { SearchOutlined } from "@ant-design/icons";
import Highlighter from "react-highlight-words";
import {
    generateLabelOptions,
    generatePropertyOptions,
    generateUrl,
    mapLabelColor,
} from "./Utils";

const formItemLayout = {
    labelCol: { span: 6 },
    wrapperCol: { span: 14 },
};

const generateDataSource = (data) => {
    return data.map((obj, idx) => ({ ...obj, key: idx }));
};

function AdvancedSearch(props) {
    const [searchInput, setSearchInput] = useState("");
    const [searchText, setSearchText] = useState("");
    const [searchedColumn, setSearchedColumn] = useState("");
    const [form] = Form.useForm();

    const onInputChange = (e) => {
        props.setSearchUserInput(e.target.value);
    };

    const onLabelChange = (value) => {
        props.setFilterLabels(value);
    };

    const onPropertiesChange = (value) => {
        props.setFilterProperties(value);
    };

    const onReset = () => {
        form.resetFields();
        props.setSearchUserInput("");
        props.setFilterLabels([]);
        props.setFilterProperties([]);
    };

    const onFinish = () => {
        props.handleAdvancedSearch();
    };

    const dataSource =
        props.matchingNodes && generateDataSource(props.matchingNodes);

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
            if (dataIndex === "period") {
                if (Number.isInteger(parseInt(value))) {
                    return Array.isArray(record.period) &&
                        record.period.length === 2
                        ? parseInt(value) >= record.period[0] &&
                              parseInt(value) <= record.period[1]
                        : "";
                } else {
                    return record.period
                        ? record.period
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
            title: "Label",
            dataIndex: "label",
            key: "label",
            filters: props.matchingNodes
                ? Array.from(
                      new Set(
                          props.matchingNodes.map((obj) => obj.label.toString())
                      )
                  ).map((value) => {
                      return { text: value, value: value };
                  })
                : [],
            filterMode: "tree",
            filterSearch: true,
            onFilter: (value, record) =>
                record.label
                    ? record.label
                          .toString()
                          .toLowerCase()
                          .includes(value.toLowerCase())
                    : false,
            render: (entry, record, idx) => {
                return (
                    <Tag
                        key={idx}
                        color={mapLabelColor(record.label.toString())}
                    >
                        {entry}
                    </Tag>
                );
            },
            sorter: (a, b) =>
                a.label.toString().localeCompare(b.label.toString()),
        },
        {
            title: "Name",
            dataIndex: "name",
            key: "name",
            ...getColumnSearchProps("name"),
            render: (entry) => (
                <Tooltip
                    placement="topLeft"
                    title="Click on the name to select the entry and examine it in more detail."
                >
                    {entry.toString().slice(0, -3)}
                </Tooltip>
            ),
            sorter: (a, b) =>
                a.name.toString().localeCompare(b.name.toString()),
            onCell: (record) => {
                return {
                    onClick: () => {
                        props.setSelectedUri(record.uri);
                        //props.setSelectedLabel(record.label)
                    },
                };
            },
        },
        {
            title: "Subcategories",
            dataIndex: "subcategories",
            key: "subcategories",
            filters: props.matchingNodes
                ? Array.from(
                      new Set(
                          props.matchingNodes
                              .map((obj) => obj.subcategories)
                              .flat()
                      )
                  ).map((value) => {
                      return { text: value.slice(4), value: value };
                  })
                : [],
            filterMode: "tree",
            filterSearch: true,
            onFilter: (value, record) =>
                record.subcategories
                    ? record.subcategories
                          .toString()
                          .toLowerCase()
                          .includes(value.toLowerCase())
                    : false,
            render: (entry) => {
                if (Array.isArray(entry)) {
                    return entry.map((value, idx) => (
                        <Tag key={idx}>{value.slice(4)}</Tag>
                    ));
                } else {
                    return entry;
                }
            },
        },
        {
            title: "Period",
            dataIndex: "period",
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
                a.period.toString().localeCompare(b.period.toString()),
        },
        {
            title: "Locations",
            dataIndex: "locations",
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
    ];

    return (
        <>
            <Form
                form={form}
                name="advancedSearch"
                {...formItemLayout}
                onFinish={onFinish}
            >
                <Form.Item
                    {...formItemLayout}
                    name="searchTerm"
                    label="Search by term"
                    tooltip="Input your search term"
                >
                    <Input
                        placeholder="Input your search term"
                        onChange={onInputChange}
                        allowClear={true}
                    />
                </Form.Item>
                <Form.Item
                    label="Filter labels"
                    name="labelsFilter"
                    tooltip="Select one or multiple labels"
                >
                    <Select
                        mode="multiple"
                        placeholder="Select one or multiple labels"
                        onChange={onLabelChange}
                        allowClear={true}
                    >
                        {props.labels && generateLabelOptions(props.labels)})
                    </Select>
                </Form.Item>
                <Form.Item
                    label="Filter properties"
                    name="propertiesFilter"
                    tooltip="Select one or multiple properties"
                >
                    <Select
                        mode="multiple"
                        placeholder="Select one or multiple properties"
                        onChange={onPropertiesChange}
                        allowClear={true}
                    >
                        {props.properties &&
                            generatePropertyOptions(props.properties)}
                        )
                    </Select>
                </Form.Item>
                <Form.Item
                    name="controllButtons"
                    wrapperCol={{ offset: 6, span: 14 }}
                >
                    <Row justify="center">
                        <Col span={4}>
                            <Button
                                type="link"
                                size="small"
                                shape="default"
                                style={{ width: "100%" }}
                                htmlType="button"
                                onClick={onReset}
                            >
                                Reset
                            </Button>
                        </Col>
                        <Col span={4} offset={12}>
                            <Badge
                                count={
                                    props.searchCount ? props.searchCount : 0
                                }
                                showZero={true}
                                overflowCount={999}
                                offset={[15, -5]}
                                title="Number of reachable nodes"
                            >
                                <Button
                                    type="primary"
                                    size="small"
                                    shape="default"
                                    style={{ width: "100%" }}
                                    htmlType="submit"
                                >
                                    Start Search
                                </Button>
                            </Badge>
                        </Col>
                    </Row>
                </Form.Item>
            </Form>
            <Divider className="Divider" orientation="center">
                Select a node to start investigation
            </Divider>
            <Table dataSource={dataSource} columns={columns} />
        </>
    );
}

export default AdvancedSearch;
