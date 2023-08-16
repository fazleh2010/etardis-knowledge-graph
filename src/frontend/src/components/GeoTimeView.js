import React, { useEffect, useState } from "react";

import { Empty, Slider, Space } from "antd";
import AnyChart from "anychart-react";
import anychart from "anychart";
import { mapLabelColor } from "./Utils";

import "../App.css";

const createMapEntry = (properties) => {
    const mappedNode = [];
    properties.locations &&
        Array.isArray(properties.locations) &&
        Array.isArray(properties.period) &&
        properties.locations.forEach((loc) => {
            mappedNode.push({
                id: properties.uri.toString(),
                name:
                    properties.name.toString().slice(-3) === "@en"
                        ? properties.name.toString().slice(0, -3)
                        : properties.name.toString(),
                label: properties.label.toString(),
                fill: mapLabelColor(properties.label.toString()),
                subcategories: Array.isArray(properties.subcategories)
                    ? properties.subcategories.join(", ")
                    : properties.subcategories,
                period: properties.period,
                lat: loc[0], // TODO: check if correct
                long: loc[1],
            });
        });
    return mappedNode;
};

const handleRelations = (relations) => {
    const mappedRelations = [];
    relations &&
        relations.forEach((rel) => {
            rel.target && mappedRelations.push(...createMapEntry(rel.target));
        });
    return mappedRelations;
};

const createMapData = (properties, relations) => {
    const mapData = [];
    properties && mapData.push(...createMapEntry(properties));
    relations && mapData.push(...handleRelations(relations));
    return mapData;
};

const calculateMarks = (minYear, maxYear) => {
    let marks = {};
    let difference = maxYear - minYear;
    for (var idx = 0; idx <= 10; idx++) {
        marks[idx * 10] = (minYear + (idx / 10.0) * difference) | 0;
    }
    return marks;
};

const calculateDateRange = (minYear, maxYear, dateRange) => {
    let difference = maxYear - minYear;
    return [
        (minYear + difference * (dateRange[0] / 100)) | 0,
        (maxYear - (difference - difference * (dateRange[1] / 100))) | 0,
    ];
};

function GeoTimeView(props) {
    const [mapData, setMapData] = useState([]);
    const [minimumYear, setMinimumYear] = useState(0);
    const [maximumYear, setMaximumYear] = useState(2022);
    const [marks, setMarks] = useState({ 0: 0, 100: 2022 });
    const [selectedDateRange, setSelectedDateRange] = useState([0, 2022]);

    useEffect(() => {
        props.nodeProperties &&
            props.nodeRelations &&
            setMapData(
                createMapData(props.nodeProperties, props.nodeRelations)
            );
    }, [props.nodeProperties, props.nodeRelations]);

    useEffect(() => {
        if (Array.isArray(mapData) && mapData.length > 0) {
            const filteredData = mapData.filter((entry) => {
                return entry.period !== "unknown";
            });
            if (filteredData.length > 0) {
                let minYear = filteredData[0].period[0];
                let maxYear = filteredData[0].period[1];
                filteredData.forEach((entry) => {
                    if (entry.period[0] < minYear) {
                        minYear = entry.period[0];
                    }
                    if (entry.period[1] > maxYear) {
                        maxYear = entry.period[1];
                    }
                });
                setMinimumYear(minYear);
                setMaximumYear(maxYear);
                setMarks(calculateMarks(minYear, maxYear));
            }
        }
    }, [mapData]);

    const map = anychart.map();
    map.geoData("anychart.maps.world");

    // create dot representation for each data entry
    props &&
        props.labels &&
        props.labels.forEach((lbl) => {
            let color = mapLabelColor(lbl);
            var series = map.marker(
                mapData.filter((entry) => {
                    return (
                        entry.period[0] >= selectedDateRange[0] &&
                        entry.period[1] <= selectedDateRange[1]
                    );
                })
            );
            series
                .name(lbl)
                .fill(color)
                .stroke("2 #E1E1E1")
                .type("circle")
                .size(10)
                .labels(false)
                .selectionMode("none");

            series.hovered().stroke("2 #fff").size(8);

            series
                .legendItem()
                .iconType("circle")
                .iconFill(color)
                .iconStroke("2 #E1E1E1");
        });

    // configure color of world map background
    map.unboundRegions().enabled(true).fill("#E1E1E1").stroke("#D2D2D2");

    // sets chart title
    map.title()
        .enabled(true)
        .useHtml(true)
        .padding([20, 0, 10, 0])
        .text("Geological and time sensitive representation");

    // configure settings for map tooltip
    map.tooltip().title().fontColor("#fff");
    map.tooltip().titleFormat("{%name}");
    map.tooltip()
        .useHtml(true)
        .padding([8, 13, 10, 13])
        .width(350)
        .fontSize(12)
        .fontColor("#e6e6e6")
        .format((entry) => {
            return (
                '<span style="font-size: 13px"><span style="color: #bfbfbf">Label: </span>' +
                entry.getData("label") +
                "<br/>" +
                '<span style="color: #bfbfbf">Subcategories: </span>' +
                entry.getData("subcategories") +
                "<br/>" +
                '<span style="color: #bfbfbf">Period: </span>' +
                entry.getData("period").join(" - ") +
                "<br/>" +
                '<span style="color: #bfbfbf">URI: </span>' +
                entry.getData("id") +
                "</span>"
            );
        });

    // turns on the legend for the sample
    map.legend(true);

    // create zoom controls
    var zoomController = anychart.ui.zoom();
    zoomController.target(map);
    zoomController.render();

    return props.nodeProperties ? (
        <Space direction="vertical">
            <AnyChart
                id="map-container"
                instance={map}
                width={1400}
                height={700}
            />
            <Slider
                range={{ draggableTrack: true }}
                marks={marks}
                defaultValue={[0, 100]}
                tipFormatter={(value) =>
                    (minimumYear +
                        (maximumYear - minimumYear) * (value / 100)) |
                    0
                }
                //tooltipVisible={true}
                onAfterChange={(value) => {
                    setSelectedDateRange(
                        calculateDateRange(minimumYear, maximumYear, value)
                    );
                }}
            />
        </Space>
    ) : (
        <Empty />
    );
}

export default GeoTimeView;
