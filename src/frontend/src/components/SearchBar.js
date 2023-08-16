import React from "react";
import { Tabs } from "antd";
import { SearchOutlined, ExperimentOutlined } from "@ant-design/icons";
import AdvancedSearch from "./AdvancedSearch";
import SimpleSearch from "./SimpleSearch";
const { TabPane } = Tabs;

function SearchBar(props) {
    return (
        <Tabs defaultActiveKey="1">
            <TabPane
                tab={
                    <span>
                        <SearchOutlined />
                        Simple search
                    </span>
                }
                key="1"
            >
                <SimpleSearch
                    labels={props.labels}
                    selectedLabel={props.selectedLabel}
                    setSelectedLabel={props.setSelectedLabel}
                    selectedUri={props.selectedUri}
                    setSelectedUri={props.setSelectedUri}
                    setUriUserInput={props.setUriUserInput}
                    matchingUris={props.matchingUris}
                />
            </TabPane>
            <TabPane
                tab={
                    <span>
                        <ExperimentOutlined />
                        Advanced search
                    </span>
                }
                key="2"
            >
                <AdvancedSearch
                    statistics={props.statistics}
                    labels={props.labels}
                    properties={props.properties}
                    filterProperties={props.filterProperties}
                    setSearchUserInput={props.setSearchUserInput}
                    setFilterLabels={props.setFilterLabels}
                    setFilterProperties={props.setFilterProperties}
                    searchCount={props.searchCount}
                    matchingNodes={props.matchingNodes}
                    handleAdvancedSearch={props.handleAdvancedSearch}
                    setSelectedLabel={props.setSelectedLabel}
                    setSelectedUri={props.setSelectedUri}
                />
            </TabPane>
        </Tabs>
    );
}

export default SearchBar;
