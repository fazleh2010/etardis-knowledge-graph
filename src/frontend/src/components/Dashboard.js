import React, { useEffect, useState } from "react";
import { Empty, Space } from "antd";
import AnyChart from "anychart-react";
import anychart from "anychart";
import { mapLabelColor } from "./Utils";

const createPieChartData = (data) => {
    const pieData = {};
    Array.isArray(data) &&
        data.forEach((entry) => {
            if (pieData[entry.target.label.toString()]) {
                pieData[entry.target.label.toString()] += 1;
            } else {
                pieData[entry.target.label.toString()] = 1;
            }
        });
    return Object.keys(pieData).map((key) => {
        return { x: key, value: pieData[key], fill: mapLabelColor(key) };
    });
};

const calculateDateRange = (data) => {
    let minYear = 0;
    let maxYear = 2022;
    if (Array.isArray(data) && data.length > 0) {
        const filteredData = data.filter((entry) => {
            return entry.target.period !== "unknown";
        });
        if (filteredData.length > 0) {
            minYear = filteredData[0].target.period[0];
            maxYear = filteredData[0].target.period[1];
            filteredData.forEach((entry) => {
                if (entry.target.period[0] < minYear) {
                    minYear = entry.target.period[0];
                }
                if (entry.target.period[1] > maxYear) {
                    maxYear = entry.target.period[1];
                }
            });
        }
    }
    return { minYear, maxYear };
};

const createBarChartData = (data) => {
    const barData = { unknown: 0 };

    const { minYear, maxYear } = calculateDateRange(data);
    for (var idx = minYear; idx <= maxYear; idx++) {
        barData[idx] = 0;
    }

    data.forEach((entry) => {
        if (Array.isArray(entry.target.period)) {
            let [start, stop] = entry.target.period;
            for (var idx = start; idx <= stop; idx++) {
                barData[idx] = barData[idx] + 1;
            }
        } else {
            barData["unknown"] = barData["unknown"] + 1;
        }
    });

    return Object.keys(barData).map((key) => {
        return {
            x: key,
            value: barData[key],
            fill: key === "unknown" ? "#8C0343" : "#66A3D9",
            label:
                key === "unknown"
                    ? {
                          enabled: true,
                          format: "{%x}",
                          position: "right-top",
                      }
                    : { enabled: false },
        };
    });
};

const createWordCloudChartData = (data) => {
    const wordCloudData = {};
    Array.isArray(data) &&
        data.forEach((entry) => {
            Object.values(entry.properties).forEach((value) => {
                if (wordCloudData[value.toString()]) {
                    wordCloudData[value.toString()] += 1;
                } else {
                    wordCloudData[value.toString()] = 1;
                }
            });
        });
    return Object.keys(wordCloudData).map((key) => {
        return { x: key, value: wordCloudData[key] };
    });
};

function Dashboard(props) {
    const [pieData, setPieData] = useState([]);
    const [barData, setBarData] = useState([]);
    const [wordCloudData, setWordCloudData] = useState([]);

    useEffect(() => {
        if (props.nodeRelations) {
            setPieData(createPieChartData(props.nodeRelations));
            setBarData(createBarChartData(props.nodeRelations));
            setWordCloudData(createWordCloudChartData(props.nodeRelations));
        }
    }, [props.nodeRelations]);

    // create a pie chart and set the data
    const pieChart = anychart.pie(pieData);
    pieChart.title("Distribution of relationship labels");
    pieChart.innerRadius("20%");

    // create a bar chart and set the data
    const barChart = anychart.column(barData);
    barChart.title("Distribution of relations over time");
    barChart.xAxis().title("Year");
    barChart.yAxis().title("Frequency of occurrence");
    barChart.xScroller().enabled(true);
    barChart.yScale().minimum(0);
    barChart.yScale().ticks().allowFractional(false);

    const wordCloudChart = anychart.tagCloud(wordCloudData);
    wordCloudChart.title("Distribution of relation properties");
    wordCloudChart.angles([0]);
    wordCloudChart.mode("spiral");
    // set the color scale as the color scale of the chart
    wordCloudChart.colorScale(
        anychart.scales.linearColor().colors(["#F29325", "#0099DD"])
    );
    wordCloudChart.colorRange().enabled(true);
    wordCloudChart.colorRange().length("75%");

    return props.nodeRelations ? (
        <Space align="baseline" size="large">
            <AnyChart
                id="pie-container"
                instance={pieChart}
                width={500}
                height={400}
            />
            <AnyChart
                id="bar-container"
                instance={barChart}
                width={500}
                height={400}
            />
            <AnyChart
                id="word-container"
                instance={wordCloudChart}
                width={500}
                height={400}
            />
        </Space>
    ) : (
        <Empty />
    );
}

export default Dashboard;
