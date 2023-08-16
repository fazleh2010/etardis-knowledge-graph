import React, { useEffect, useState } from "react";
import {
    Badge,
    Button,
    Col,
    Collapse,
    Descriptions,
    Empty,
    Input,
    Row,
    Space,
    Table,
    Tag,
    Tooltip,
} from "antd";
import { InfoCircleOutlined, SearchOutlined } from "@ant-design/icons";
import Highlighter from "react-highlight-words";
import { v4 as uuid } from "uuid";
import { generateUrl, mapLabelColor } from "./Utils";
const { Panel } = Collapse;

const formatEntry = (value, index) => {
    if (value && value.toString().includes("http")) {
        return generateUrl(value, value, index);
    } else if (value && value.toString().slice(-3) === "@en") {
        return value.toString().slice(0, -3);
    } else {
        return value.toString();
    }
};

const handleValues = (values) => {
    if (values && Array.isArray(values) && values.length > 1) {
        return values.map((val, idx) => {
            return <p key={"val_" + idx}>{formatEntry(val, idx)}</p>;
        });
    } else {
        return formatEntry(values, uuid().slice(0, 8));
    }
};

const handleDescriptionKey = (key) => {
    if (key === "locations") {
        return "available locations";
    } else {
        return key;
    }
};

const handleDescriptionInfo = (value) => {
    switch (value) {
        case "uri":
            return "uri text.";
        case "name":
            return "name text.";
        case "period":
            return "period text.";
        case "locations":
            return "locations text.";
        case "label":
            return "label text.";
        case "subcategories":
            return "subcategories text.";
        case "facets":
            return "facets text.";
        case "participants":
            return "participants text.";
        case "families":
            return "families text.";
        case "genders":
            return "genders text.";
        case "positions":
            return "positions text.";
        case "religions":
            return "religions text.";
        case "authors":
            return "authors text.";
        case "material":
            return "material text.";
        case "uses":
            return "uses text.";
        default:
            return "Some text.";
    }
};

const generateDescription = (properties) => {
    const propertyContent = [];
    propertyContent.push(
        properties &&
            Object.keys(properties)
                .filter((k) => k !== "properties")
                .map((k, idx) => {
                    let value;
                    if (k === "label") {
                        value = (
                            <Tag
                                color={mapLabelColor(
                                    handleValues(properties[k])
                                )}
                            >
                                {handleValues(properties[k])}
                            </Tag>
                        );
                    } else if (k === "period") {
                        value = Array.isArray(properties[k])
                            ? properties[k].join(" - ")
                            : properties[k];
                    } else if (k === "subcategories") {
                        value = Array.isArray(properties[k])
                            ? properties[k].map((value, idx) => (
                                  <Tag key={idx}>{value.slice(4)}</Tag>
                              ))
                            : properties[k];
                    } else if (k === "locations") {
                        value = Array.isArray(properties[k])
                            ? properties[k].map((loc, jdx) => {
                                  return (
                                      <Tag
                                          key={
                                              idx.toString +
                                              "_" +
                                              jdx.toString()
                                          }
                                          color="processing"
                                      >
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
                                              "(" +
                                                  loc[0] +
                                                  ", " +
                                                  loc[1] +
                                                  ")",
                                              idx + "_" + jdx
                                          )}
                                      </Tag>
                                  );
                              })
                            : properties[k];
                    } else {
                        value = handleValues(properties[k]);
                    }
                    return (
                        <Descriptions.Item
                            key={idx}
                            label={
                                <Tooltip
                                    title={handleDescriptionInfo(k)}
                                    placement="right"
                                >
                                    <Badge
                                        count={<InfoCircleOutlined />}
                                        offset={[10, 0]}
                                    >
                                        {handleDescriptionKey(k)}
                                    </Badge>
                                </Tooltip>
                            }
                        >
                            {value}
                        </Descriptions.Item>
                    );
                })
    );
    return propertyContent;
};

const generateDataSource = (properties) => {
    const dataSource = [];
    properties &&
        properties.properties &&
        Object.keys(properties.properties).forEach((k, idx) => {
            dataSource.push({
                key: idx,
                property: k,
                value: handleValues(properties.properties[k]),
            });
        });
    return dataSource;
};

function DescriptionTable(props) {
    const [description, setDescription] = useState([]);
    const [dataSource, setDataSource] = useState([]);
    const [searchInput, setSearchInput] = useState("");
    const [searchText, setSearchText] = useState("");
    const [searchedColumn, setSearchedColumn] = useState("");

    useEffect(() => {
        props.nodeProperties &&
            setDataSource(generateDataSource(props.nodeProperties));
        props.nodeProperties &&
            setDescription(generateDescription(props.nodeProperties));
    }, [props.nodeProperties]);

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
            return record[dataIndex]
                ? record[dataIndex]
                      .toString()
                      .toLowerCase()
                      .includes(value.toLowerCase())
                : "";
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
            title: "Property",
            dataIndex: "property",
            key: "property",
            filters: dataSource
                ? [...new Set(dataSource.map((entry) => entry.property))].map(
                      (value) => {
                          return {
                              text: value,
                              value: value,
                          };
                      }
                  )
                : [],
            filterMode: "tree",
            filterSearch: true,
            onFilter: (value, record) =>
                record.property
                    ? record.property
                          .toLowerCase()
                          .includes(value.toLowerCase())
                    : false,
            sorter: (a, b) => a.property.localeCompare(b.property),
        },
        {
            title: "Value",
            dataIndex: "value",
            key: "value",
            ...getColumnSearchProps("value"),
        },
    ];

    return dataSource ? (
        <Space direction="vertical">
            <Descriptions
                title="Description"
                size="small"
                column={1}
                bordered={true}
            >
                {description}
            </Descriptions>
            <Collapse accordion={true}>
                <Panel header="Further properties" key="1">
                    <Table dataSource={dataSource} columns={columns} />
                </Panel>
            </Collapse>
        </Space>
    ) : (
        <Empty />
    );
}

export default DescriptionTable;
