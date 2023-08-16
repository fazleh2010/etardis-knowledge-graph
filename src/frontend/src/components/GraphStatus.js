import React from "react";
import { Row, Col, Statistic } from "antd";

function GraphStatus(props) {
    return (
        <Row gutter={12} justify="space-between">
            <Col span={4}>
                <Statistic
                    title="Graph Status"
                    value={
                        props.statistics.available
                            ? "Available"
                            : "Not available"
                    }
                />
            </Col>
            <Col span={4}>
                <Statistic
                    title="Total nodes"
                    value={props.statistics.totalNodes}
                />
            </Col>
            <Col span={4}>
                <Statistic
                    title="Total relations"
                    value={props.statistics.totalRelations}
                />
            </Col>
        </Row>
    );
}

export default GraphStatus;
