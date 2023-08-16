import React, { useState, useEffect } from "react";
import { Badge, Row, Col, Divider, Tooltip, Popover } from "antd";
import SearchBar from "./components/SearchBar";
import DescriptionTable from "./components/DescriptionTable";
import Dashboard from "./components/Dashboard";
import RelationTable from "./components/RelationTable";
import GeoTimeView from "./components/GeoTimeView";
import GraphStatus from "./components/GraphStatus";
import { InfoCircleOutlined } from "@ant-design/icons";

import "./App.css";

function App() {
    const [labels, setLabels] = useState([]);
    const [properties, setProperties] = useState([]);
    // advanced search
    const [searchUserInput, setSearchUserInput] = useState("");
    const [filterLabels, setFilterLabels] = useState([]);
    const [filterProperties, setFilterProperties] = useState([]);
    const [searchCount, setSearchCount] = useState(0);
    const [matchingNodes, setMatchingNodes] = useState([]);
    // simple search
    const [selectedLabel, setSelectedLabel] = useState("");
    const [selectedUri, setSelectedUri] = useState("");
    const [uriUserInput, setUriUserInput] = useState("");
    const [matchingUris, setMatchingUris] = useState([]);
    // node content
    const [nodeProperties, setNodeProperties] = useState();
    const [nodeRelations, setNodeRelations] = useState();
    const [statistics, setStatistics] = useState({
        available: false,
        totalNodes: 0,
        totalRelations: 0,
    });

    const placeholderText =
        "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.";

    // proxy handles development environment
    const apiUrl =
        process.env.REACT_APP_API_URL && process.env.REACT_APP_API_PORT
            ? `http://${process.env.REACT_APP_API_URL}:${process.env.REACT_APP_API_PORT}`
            : "";

    useEffect(() => {
        // call init once when page is loaded
        handleGraphStatus();
        handleLabels();
        handleProperties();
    }, []);

    useEffect(() => {
        handleSimpleSearch();
    }, [uriUserInput]);

    useEffect(() => {
        if (selectedUri) {
            handleNodeProperties();
            handleNodeRelations();
        }
    }, [selectedUri]);

    useEffect(() => {
        handleAdvancedSearchCounter();
    }, [searchUserInput, filterLabels, filterProperties]);

    /**
     * It fetches the statistics from the server and sets the statistics object to the data returned by
     * the server.
     */
    function handleGraphStatus() {
        fetch(`${apiUrl}/getStatistics`, {
            method: "GET",
        })
            .then((res) => res.json())
            .then((data) => {
                setStatistics({
                    available: data.available,
                    totalNodes: data.totalNodes,
                    totalRelations: data.totalRelations,
                });
            })
            .catch((error) => console.log("ERROR statistics: " + error));
    }

    /**
     * Fetch the labels from the server and store them in the state.
     */
    function handleLabels() {
        fetch(`${apiUrl}/getLabels`, {
            method: "GET",
        })
            .then((res) => res.json())
            .then((data) => {
                setLabels(data.labels);
            })
            .catch((error) =>
                console.log("ERROR getting available labels: " + error)
            );
    }

    function handleProperties() {
        fetch(`${apiUrl}/getProperties`, {
            method: "GET",
        })
            .then((res) => res.json())
            .then((data) => {
                setProperties(data.properties);
            })
            .catch((error) =>
                console.log("ERROR getting available properties: " + error)
            );
    }

    function handleAdvancedSearchCounter() {
        fetch(`${apiUrl}/getNodesCountByInput`, {
            headers: {
                "Content-Type": "application/json",
            },
            method: "POST",
            body: JSON.stringify({
                userInput: searchUserInput,
                filteredLabels: filterLabels,
                filteredProperties: filterProperties,
            }),
        })
            .then((res) => res.json())
            .then((data) => {
                setSearchCount(data.counter);
            })
            .catch((error) => {
                console.log(
                    "ERROR getting count for advanced search nodes: " + error
                );
            });
    }

    function handleAdvancedSearch() {
        fetch(`${apiUrl}/getNodesByInput`, {
            headers: {
                "Content-Type": "application/json",
            },
            method: "POST",
            body: JSON.stringify({
                userInput: searchUserInput,
                filteredLabels: filterLabels,
                filteredProperties: filterProperties,
            }),
        })
            .then((res) => res.json())
            .then((data) => {
                setMatchingNodes(data.matchingNodes);
            })
            .catch((error) =>
                console.log("ERROR getting nodes by advanced search: " + error)
            );
    }

    /**
     * It sends a POST request to the server to get the matching URIs for the user input.
     */
    function handleSimpleSearch() {
        fetch(`${apiUrl}/getMatchingUris`, {
            headers: {
                "Content-Type": "application/json",
            },
            method: "POST",
            body: JSON.stringify({
                selectedLabel: selectedLabel,
                userInput: uriUserInput,
            }),
        })
            .then((res) => res.json())
            .then((data) => {
                setMatchingUris(data.matchingUris);
            })
            .catch((error) => console.log("ERROR get matching uris: " + error));
    }

    /**
     * It fetches the properties of the selected node.
     */
    function handleNodeProperties() {
        fetch(`${apiUrl}/getSingleNode`, {
            headers: {
                "Content-Type": "application/json",
            },
            method: "POST",
            body: JSON.stringify({
                uri: selectedUri,
            }),
        })
            .then((res) => res.json())
            .then((data) => {
                setNodeProperties(data.node);
            })
            .catch((error) => console.log("ERROR get single node: " + error));
    }

    /**
     * It fetches the relations of the selected node from the database.
     */
    function handleNodeRelations() {
        fetch(`${apiUrl}/getConnectionsForNode`, {
            headers: {
                "Content-Type": "application/json",
            },
            method: "POST",
            body: JSON.stringify({
                uri: selectedUri,
                threshold: 0,
            }),
        })
            .then((res) => res.json())
            .then((data) => {
                setNodeRelations(data.relations);
            })
            .catch((error) =>
                console.log("ERROR get connections for node: " + error)
            );
    }

    return (
        <div className="App">
            <header className="App-header">
                <Popover
                    title="General information"
                    overlayInnerStyle={{ width: "80%", margin: "0 auto" }}
                    content={
                        <div>
                            <p>{placeholderText}</p>
                        </div>
                    }
                >
                    <Badge
                        style={{ justifyContent: "center" }}
                        count={<InfoCircleOutlined />}
                        offset={[10, 20]}
                    >
                        <h1 className="h1">eTaRDiS Knowledge Graph Explorer</h1>
                    </Badge>
                </Popover>
                <div className="placeholder-text">
                    <p>{placeholderText}</p>
                </div>
            </header>
            <span className="App-body">
                <Row justify="center" gutter={[24, 24]}>
                    <Col span={20}>
                        <SearchBar
                            // simple search
                            labels={labels}
                            selectedLabel={selectedLabel}
                            setSelectedLabel={setSelectedLabel}
                            selectedUri={selectedUri}
                            setSelectedUri={setSelectedUri}
                            setUriUserInput={setUriUserInput}
                            matchingUris={matchingUris}
                            // advanced search
                            statistics={statistics}
                            properties={properties}
                            filterProperties={filterProperties}
                            setSearchUserInput={setSearchUserInput}
                            setFilterLabels={setFilterLabels}
                            setFilterProperties={setFilterProperties}
                            searchCount={searchCount}
                            matchingNodes={matchingNodes}
                            handleAdvancedSearch={handleAdvancedSearch}
                        />
                    </Col>
                </Row>
                <Divider className="Divider" orientation="center">
                    <Popover
                        title="Properties information"
                        overlayInnerStyle={{ width: "80%", margin: "0 auto" }}
                        content={
                            <div>
                                <p>{placeholderText}</p>
                            </div>
                        }
                    >
                        <Badge count={<InfoCircleOutlined />} offset={[10, 0]}>
                            Explore properties
                        </Badge>
                    </Popover>
                </Divider>
                <div className="placeholder-text">
                    <p>{placeholderText}</p>
                </div>
                <Row justify="center" gutter={[24, 24]}>
                    <Col span={20}>
                        <DescriptionTable nodeProperties={nodeProperties} />
                    </Col>
                </Row>
                <Divider className="Divider" orientation="center">
                    <Popover
                        title="Relations information"
                        overlayInnerStyle={{ width: "80%", margin: "0 auto" }}
                        content={
                            <div>
                                <p>{placeholderText}</p>
                            </div>
                        }
                    >
                        <Badge count={<InfoCircleOutlined />} offset={[10, 0]}>
                            Explore relations
                        </Badge>
                    </Popover>
                </Divider>
                <div className="placeholder-text">
                    <p>{placeholderText}</p>
                </div>
                <Row justify="center" gutter={[24, 24]}>
                    <Col span={20}>
                        <RelationTable
                            nodeRelations={nodeRelations}
                            setSelectedUri={setSelectedUri}
                            labels={labels}
                        />
                    </Col>
                </Row>
                <Divider className="Divider" orientation="center">
                    <Popover
                        title="Dashboard information"
                        overlayInnerStyle={{ width: "80%", margin: "0 auto" }}
                        content={
                            <div>
                                <p>{placeholderText}</p>
                            </div>
                        }
                    >
                        <Badge count={<InfoCircleOutlined />} offset={[10, 0]}>
                            Dashboard
                        </Badge>
                    </Popover>
                </Divider>
                <div className="placeholder-text">
                    <p>{placeholderText}</p>
                </div>
                <Row justify="center" gutter={[24, 24]}>
                    <Col span={20}>
                        <Dashboard nodeRelations={nodeRelations} />
                    </Col>
                </Row>
                <Row justify="center" gutter={[24, 24]}>
                    <Col span={20}>
                        <GeoTimeView
                            nodeProperties={nodeProperties}
                            nodeRelations={nodeRelations}
                            labels={labels}
                        />
                    </Col>
                </Row>
                <Divider className="Divider" orientation="center">
                    <Tooltip title="Some text.">
                        <Badge count={<InfoCircleOutlined />} offset={[10, 0]}>
                            Graph status
                        </Badge>
                    </Tooltip>
                </Divider>
                <Row justify="center" gutter={[24, 24]}>
                    <Col span={20}>
                        <GraphStatus statistics={statistics} />
                    </Col>
                </Row>
            </span>
        </div>
    );
}

export default App;
