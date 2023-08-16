import React from "react";
import { Select } from "antd";
const { Option } = Select;

export const mapLabelColor = (label) => {
    switch (label) {
        case "Agent":
            return "#D8B08C";
        case "CulturalArtifact":
            return "#45C4B0";
        case "Event":
            return "#13678A";
        case "MaterialObject":
            return "#E1523D";
        case "Place":
            return "#9AEBA3";
        case "TimePeriod":
            return "#D2E8E3";
        case "TopicalConcept":
            return "#A6445E";
        default:
            return "#012030";
    }
};

export const generateUrl = (url, content, idx) => {
    return (
        <a
            key={"url_" + idx}
            href={url}
            target="_blank"
            rel="noopener noreferrer"
        >
            {content}
        </a>
    );
};

export const generateLabelOptions = (values) => {
    return values.map((x) => <Option key={x}>{x}</Option>);
};

export const generatePropertyOptions = (values) => {
    return values.map((x) => <Option key={x.uri}>{x.formattedUri}</Option>);
};
