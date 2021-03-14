# The dashboard is visible by clicking on http://127.0.0.1:8085/
# Import packages
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from pandas import read_csv
from Modules.config import *
import os
from Dashboard.elaboration import *

# Retrieve original dataframe
original_dataframe = read_csv(os.path.join(base_dir, 'HomeC.csv'), sep=',')
# Dataframe preprocessed once for all the pictures
dataframe = hard_process_dataframe(original_dataframe)
# Preprocess the dataframe without combining some features
light_dataframe = light_process_dataframe(original_dataframe)
# List of features needed to the dropdown object
list_1 = dataframe.columns

# Style arguments for the sidebar and the content page.
SIDEBAR_STYLE = {'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0, 'width': '20%', 'padding': '20px 10px',
                 'background-color': '#f8f9fa'}
CONTENT_STYLE = {'margin-left': '25%', 'margin-right': '5%', 'padding': '20px 10p'}
TITLE_STYLE = {'textAlign': 'center', 'color': '#191970', 'padding-top': '20px'}
TEXT_STYLE = {'textAlign': 'center', 'color': '#191970'}
CARD_TEXT_STYLE = {'textAlign': 'center', 'color': '#0074D9'}

# Define all the controls of the sidebar. It consists of a dropdown, a range slider, a slider and a submit button.
controls = dbc.FormGroup(
    [html.P('Scatter Plot', style={'textAlign': 'center'}),
     dcc.Dropdown(id='features', options=create_options(list_1), value=['Total Power', 'Pressure'], multi=True),
     html.Br(),
     html.P('Date Range', style={'textAlign': 'center'}),
     dcc.RangeSlider(id='date_range', min=0, max=len(dataframe), value=[1, len(dataframe)], step=1),
     html.P('Resampling', style={'textAlign': 'center'}),
     dcc.Slider(id='resampling', min=1, max=120, step=1, value=40),
     html.Br(),
     html.P('Distribution Plots', style={'textAlign': 'center'}),
     dcc.Dropdown(id='dist', options=create_options(list_1), value=['Total Power', 'Pressure', 'Humidity'], multi=True),
     html.Br(),
     dbc.Button(id='submit_button', n_clicks=0, children='Submit', color='primary', block=True)])

# Definition of the sidebar
sidebar = html.Div([html.H2('Parameters', style=TEXT_STYLE), html.Hr(), controls], style=SIDEBAR_STYLE)
# Definition of the rows constituting the content page
content_first_row = dbc.Row([dbc.Col(dcc.Graph(id='graph_1'), md=6), dbc.Col(dcc.Graph(id='graph_2'), md=6)])
content_second_row = dbc.Row([dbc.Col(dcc.Graph(id='graph_3'), md=6), dbc.Col(dcc.Graph(id='graph_4'), md=6)])
content_third_row = dbc.Row([dbc.Col(dcc.Graph(id='graph_5'), md=12)])
content_fourth_row = dbc.Row([dbc.Col(dcc.Graph(id='graph_6'), md=12)])
content_sixth_row = dbc.Row([dbc.Col(dcc.Graph(id='graph_7'), md=4), dbc.Col(dcc.Graph(id='graph_8'), md=4),
                             dbc.Col(dcc.Graph(id='graph_9'), md=4)])
# Definition of the three cards that display dynamically the parameters selected
content_fifth_row = dbc.Row([
    dbc.Col(dbc.Card([dbc.CardBody([
        html.H4(id='card_title_1', children=['Resampling'], className='card-title', style=CARD_TEXT_STYLE),
        html.P(id='card_text_1', children=['Sample text.'], style=CARD_TEXT_STYLE)])]), md=4),
    dbc.Col(dbc.Card([dbc.CardBody([
        html.H4(id='card_title_2', children=['Starting Date'], className='card-title', style=CARD_TEXT_STYLE),
        html.P(id='card_text_2', children=['Sample text.'], style=CARD_TEXT_STYLE), ])]),
        md=4),
    dbc.Col(dbc.Card([dbc.CardBody([
        html.H4(id='card_title_3', children=['Ending Date'], className='card-title', style=CARD_TEXT_STYLE),
        html.P(id='card_text_3', children=['Sample text.'], style=CARD_TEXT_STYLE)])]), md=4)])

# Definition of the content page
content = html.Div([html.H2('Smart Home Energy System Analytics', style=TITLE_STYLE), html.Hr(),
                    content_fifth_row, content_first_row, content_fourth_row, content_second_row, content_third_row,
                    content_sixth_row],
                   style=CONTENT_STYLE)

# Definition of the entire layout
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([sidebar, content])


@app.callback(
    Output('graph_1', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_1(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    print('Graph 1 update.' + '=' * 20)
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    # Compute the Pearson correlation matrix
    corr = df.corr()
    fig = px.imshow(corr)
    return fig


@app.callback(
    Output('graph_2', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_2(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    print('Graph 2 update.' + '=' * 20)
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    # List of the features related to the room of the house
    columns = ['Kitchen', 'Living Room', 'Furnace', 'Outside', 'Home Office']
    # Compute the pie values
    pie_values = pie_chart(df, columns)
    fig = go.Figure(data=[go.Pie(labels=columns, values=pie_values)])
    return fig


@app.callback(
    Output('graph_3', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_3(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    print('Graph 3 update.' + '=' * 20)
    # Slice the dataframe in the interval required
    df = light_dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    # List of the features related to the appliances
    columns = ['Kitchen', 'Living Room', 'Furnace', 'Home Office', 'Fridge', 'Microwave', 'Wine Cellar',
               'Dishwasher', 'Well', 'Garage Door', 'Barn']
    # Compute the pie values
    pie_values = pie_chart(df, columns)
    fig = go.Figure(data=[go.Pie(labels=columns, values=pie_values)])
    return fig


@app.callback(
    Output('graph_4', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_4(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    print('Graph 4 update.' + '=' * 20)
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    fig = px.scatter(df, x=dropdown_value[0], y=dropdown_value[1])
    return fig


@app.callback(
    Output('graph_5', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_5(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    print('Graph 5 update.' + '=' * 20)
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    fig = px.box(df)
    return fig


@app.callback(
    Output('graph_6', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_6(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    print('Graph 6 update.' + '=' * 20)
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    fig = px.line(df)
    return fig


@app.callback(
    Output('graph_7', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_7(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    print('Graph 7 update.' + '=' * 20)
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    fig = px.histogram(df, x=distribution[0], marginal="violin", hover_data=df.columns)
    return fig


@app.callback(
    Output('graph_8', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_8(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    print('Graph 8 update.' + '=' * 20)
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    fig = px.histogram(df, x=distribution[1], marginal="box", hover_data=df.columns)
    return fig


@app.callback(
    Output('graph_9', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_9(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    print('Graph 9 update.' + '=' * 20)
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    fig = px.histogram(df, x=distribution[2], marginal="rug", hover_data=df.columns)
    return fig


@app.callback(
    Output('card_text_1', 'children'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_card_text_1(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    return f'{slider_value} minutes'


@app.callback(
    Output('card_text_2', 'children'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_card_text_2(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    return f'{df.index[0]}'


@app.callback(
    Output('card_text_3', 'children'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('dist', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_card_text_3(n_clicks, dropdown_value, distribution, range_slider_value, slider_value):
    # Slice the dataframe in the interval required
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    # Resampling of the dataframe with the value required
    df = resampling_dataframe(df, f'{slider_value}min')
    return f'{df.index[-1]}'


if __name__ == '__main__':
    app.run_server(port=8085)
