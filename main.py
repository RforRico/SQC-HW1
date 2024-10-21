import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from scipy.stats import gamma, beta, binom, uniform, lognorm, f, norm

# 初始化 Dash 應用
app = dash.Dash(__name__)

# 應用程式佈局
app.layout = html.Div(
    [
        html.H1(
            "中央極限定理與樣本平均數抽樣分配",
            style={"text-align": "center", "margin-bottom": "30px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "選擇分佈:",
                            style={"font-weight": "bold", "font-size": "20px"},
                        ),
                        dcc.Dropdown(
                            id="dist-type",
                            options=[
                                {"label": "Uniform", "value": "Uniform"},
                                {"label": "Binomial", "value": "Binomial"},
                                {"label": "Gamma", "value": "Gamma"},
                                {"label": "Beta", "value": "Beta"},
                                {"label": "F", "value": "F"},
                                {"label": "Lognormal", "value": "Lognormal"},
                            ],
                            value="Uniform",
                            style={"width": "80%", "font-size": "18px"},
                        ),
                    ],
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "padding": "10px",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            id="param1-label",
                            style={"font-weight": "bold", "font-size": "20px"},
                        ),
                        dcc.Input(
                            id="param1",
                            type="number",
                            value=10,
                            min=1,
                            max=150,
                            step=1,
                            style={"font-size": "18px", "width": "100px"},
                        ),
                    ],
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "padding": "10px",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            id="param2-label",
                            style={"font-weight": "bold", "font-size": "20px"},
                        ),
                        dcc.Input(
                            id="param2",
                            type="number",
                            value=1,
                            style={"font-size": "18px", "width": "100px"},
                        ),
                    ],
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "padding": "10px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "justify-content": "space-around",
                "margin-bottom": "30px",
            },
        ),
        html.Div(
            [
                html.Label(
                    "樣本數量 (n):", style={"font-weight": "bold", "font-size": "20px"}
                ),
                dcc.Slider(
                    id="sample-size",
                    min=1,
                    max=200,
                    step=1,
                    value=30,
                    marks={i: str(i) for i in range(1, 201, 20)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"width": "80%", "margin": "auto", "margin-bottom": "30px"},
        ),
        html.Div(
            [
                html.Label(
                    "樣本平均數的個數:",
                    style={"font-weight": "bold", "font-size": "20px"},
                ),
                dcc.Input(
                    id="num-means",
                    type="number",
                    value=400,
                    min=1,
                    max=10000,
                    step=1,
                    style={"font-size": "18px", "width": "100px"},
                ),
            ],
            style={"width": "80%", "margin": "auto", "margin-bottom": "30px"},
        ),
        html.Div(
            [
                dcc.Graph(
                    id="dist-plot", style={"display": "inline-block", "width": "48%"}
                ),
                dcc.Graph(
                    id="sample-mean-plot",
                    style={"display": "inline-block", "width": "48%"},
                ),
            ]
        ),
        html.Div(id="statistics", style={"text-align": "center", "margin-top": "20px"}),
    ]
)


# 動態更新參數標籤
@app.callback(
    [
        Output("param1-label", "children"),
        Output("param2-label", "children"),
        Output("param2", "min"),
        Output("param2", "max"),
        Output("param2", "step"),
    ],
    [Input("dist-type", "value")],
)
def update_labels_and_input(dist_type):
    if dist_type == "Lognormal":
        return "μ:", "σ²:", 1, 150, 1
    elif dist_type == "F":
        return "n:", "m:", 1, 150, 1
    elif dist_type == "Binomial":
        return "n:", "p (機率):", 0, 1, 0.01
    else:
        return "α:", "β:", 1, 150, 1


# 更新圖表
@app.callback(
    [
        Output("dist-plot", "figure"),
        Output("sample-mean-plot", "figure"),
        Output("statistics", "children"),
    ],
    [
        Input("dist-type", "value"),
        Input("param1", "value"),
        Input("param2", "value"),
        Input("sample-size", "value"),
        Input("num-means", "value"),
    ],
)
def update_plots(dist_type, param1, param2, sample_size, num_means):
    try:
        x = np.linspace(0, 10, 500)

        # 母體分佈圖
        if dist_type == "Uniform":
            y = uniform.pdf(x, loc=param1, scale=param2 - param1)
        elif dist_type == "Binomial":
            x = np.arange(0, param1 + 1)
            y = binom.pmf(x, param1, param2)
        elif dist_type == "Gamma":
            y = gamma.pdf(x, param1, scale=1 / param2)
        elif dist_type == "Beta":
            x = np.linspace(0, 1, 500)
            y = beta.pdf(x, param1, param2)
        elif dist_type == "F":
            y = f.pdf(x, param1, param2)
        elif dist_type == "Lognormal":
            y = lognorm.pdf(x, s=param2, scale=np.exp(param1))

        dist_fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines")])
        dist_fig.update_layout(title="母體分佈", xaxis_title="X", yaxis_title="Density")

        # 計算樣本平均數
        means = []
        for _ in range(num_means):
            if dist_type == "Uniform":
                sample = uniform.rvs(
                    loc=param1, scale=param2 - param1, size=sample_size
                )
            elif dist_type == "Binomial":
                sample = binom.rvs(param1, param2, size=sample_size)
            elif dist_type == "Gamma":
                sample = gamma.rvs(param1, scale=1 / param2, size=sample_size)
            elif dist_type == "Beta":
                sample = beta.rvs(param1, param2, size=sample_size)
            elif dist_type == "F":
                sample = f.rvs(param1, param2, size=sample_size)
            elif dist_type == "Lognormal":
                sample = lognorm.rvs(s=param2, scale=np.exp(param1), size=sample_size)
            means.append(np.mean(sample))

        # 繪製直方圖並添加密度函數曲線
        hist, bin_edges = np.histogram(means, bins=30, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        mean_fig = go.Figure()
        mean_fig.add_trace(go.Bar(x=bin_centers, y=hist, name="直方圖", opacity=0.6))

        mean_fig.add_trace(
            go.Scatter(x=bin_centers, y=hist, mode="lines", name="密度函數")
        )

        # 添加常態曲線
        mean_mu, mean_sigma = np.mean(means), np.std(means)
        x = np.linspace(mean_mu - 4 * mean_sigma, mean_mu + 4 * mean_sigma, 100)
        y = norm.pdf(x, mean_mu, mean_sigma)

        mean_fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="常態曲線",
                line=dict(color="red", dash="dash"),
            )
        )

        mean_fig.update_layout(
            title="樣本平均數的抽樣分配", xaxis_title="Mean", yaxis_title="Density"
        )

        stats = f"樣本平均數: {mean_mu:.2f}, 標準差: {mean_sigma:.2f}"
        return dist_fig, mean_fig, stats

    except Exception as e:
        return go.Figure(), go.Figure(), f"Error: {str(e)}"


# 啟動應用
if __name__ == "__main__":
    app.run_server(debug=True)
