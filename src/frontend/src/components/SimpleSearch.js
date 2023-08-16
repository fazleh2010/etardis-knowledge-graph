import React from "react";
import { AutoComplete, Input, Select } from "antd";
import { generateLabelOptions } from "./Utils";

const { Option } = AutoComplete;

const generateUriOptions = (values) => {
    return values.map((obj) => (
        <Option
            key={obj.uri}
            value={obj.name ? obj.name.toString().slice(0, -3) : obj.uri}
        >
            {obj.name ? obj.name.toString().slice(0, -3) : obj.uri}
        </Option>
    ));
};

function SimpleSearch(props) {
    /**
     * It sets the value of the selected label to the value passed in.
     * @param value - The value of the selected option.
     */
    const onLabelChange = (value) => {
        props.setSelectedLabel(value);
    };

    /**
     * The function takes in a value from the user input and sets the uriUserInput property of the App
     * component to that value
     * @param value - The value of the input field.
     */
    const handleUserInput = (value) => {
        props.setUriUserInput(value);
    };

    /**
     * The function takes in a value from the search bar and sets the selectedUri state to that value
     * @param value - The value of the search input.
     */
    const handleSearch = (record) => {
        props.setSelectedUri(record.key);
    };

    return (
        <Input.Group compact={true}>
            <Select
                showSearch
                style={{ width: "30%" }}
                placeholder="Select a label"
                optionFilterProp="children"
                onChange={onLabelChange}
                filterOption={(input, option) =>
                    option.children
                        .toLowerCase()
                        .indexOf(input.toLowerCase()) >= 0
                }
                allowClear={true}
                //value={props.selectedLabel}
            >
                {props.labels && generateLabelOptions(props.labels)}
            </Select>
            <AutoComplete
                style={{ width: "70%" }}
                placeholder="Enter URI"
                onChange={handleUserInput}
                onSelect={(_, record) => handleSearch(record)}
                allowClear={true}
                //value={props.selectedUri}
            >
                {props.matchingUris && generateUriOptions(props.matchingUris)}
            </AutoComplete>
        </Input.Group>
    );
}

export default SimpleSearch;
