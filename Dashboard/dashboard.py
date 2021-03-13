# The dashboard is visible by clicking on http://127.0.0.1:8085/
# Import packages
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from pandas import read_csv, Timestamp, to_datetime
from Modules.config import *
import os
from Dashboard.elaboration import *
from Modules.utilities import *

# Retrieve original dataframe
dataframe = read_csv(os.path.join(rel_dir, 'HomeC.csv'), sep=',')
list_1 = ['Total Power', 'Kitchen', 'Furnace', 'Outside', 'Living Room', 'Temperature', 'Humidity', 'Pressure',
          'Wind Speed', 'Wind Bearing', 'Precipitation', 'Dew Point', 'Home Office', 'Week Day', 'Hour', 'Day', 'Month',
          'Day Of Year', 'Week Of Year', 'Day Moment']

# Style arguments for the sidebar.
SIDEBAR_STYLE = {'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0, 'width': '20%', 'padding': '20px 10px',
                 'background-color': '#f8f9fa'}
# Style arguments for the main content page.
CONTENT_STYLE = {'margin-left': '25%', 'margin-right': '5%', 'padding': '20px 10p'}
TITLE_STYLE = {'textAlign': 'center', 'color': '#191970', 'padding-top': '20px'}
TEXT_STYLE = {'textAlign': 'center', 'color': '#191970'}
CARD_TEXT_STYLE = {'textAlign': 'center', 'color': '#0074D9'}

# Below are all the controls of the sidebar which consist of a dropdown, range slider, checklist, and radio buttons.
controls = dbc.FormGroup(
    [html.P('Scatter Plot', style={'textAlign': 'center'}),
     dcc.Dropdown(id='features',
                  options=create_options(list_1),
                  value=['Total Power', 'Pressure'],  # default value
                  multi=True),
     html.Br(),
     html.P('Date Range', style={'textAlign': 'center'}),
     dcc.RangeSlider(id='date_range', min=0, max=len(dataframe), value=[1, len(dataframe)], step=1),
     html.P('Resampling', style={'textAlign': 'center'}),
     dcc.Slider(id='resampling', min=1, max=120, step=1, value=40),
     html.Br(),
     dbc.Button(id='submit_button', n_clicks=0, children='Submit', color='primary', block=True)])

sidebar = html.Div([html.H2('Parameters', style=TEXT_STYLE), html.Hr(), controls], style=SIDEBAR_STYLE)
content_first_row = dbc.Row([dbc.Col(dcc.Graph(id='graph_1'), md=6),
                             dbc.Col(dcc.Graph(id='graph_2'), md=6)])
content_second_row = dbc.Row([dbc.Col(dcc.Graph(id='graph_3'), md=6),
                              dbc.Col(dcc.Graph(id='graph_4'), md=6)])
content_third_row = dbc.Row([dbc.Col(dcc.Graph(id='graph_5'), md=12)])
content_fourth_row = dbc.Row([dbc.Col(dcc.Graph(id='graph_6'), md=12)])
content_fifth_row = dbc.Row([dbc.Col(dbc.Card([dbc.CardBody([
    html.H4(id='card_title_1', children=['Resampling'], className='card-title', style=CARD_TEXT_STYLE),
    html.P(id='card_text_1', children=['Sample text.'], style=CARD_TEXT_STYLE)])]), md=4),
    dbc.Col(dbc.Card([dbc.CardBody([
        html.H4(id='card_title_2', children=['Starting Date'], className='card-title', style=CARD_TEXT_STYLE),
        html.P(id='card_text_2', children=['Sample text.'], style=CARD_TEXT_STYLE), ])]),
        md=4),
    dbc.Col(dbc.Card([dbc.CardBody([
        html.H4(id='card_title_3', children=['Ending Date'], className='card-title', style=CARD_TEXT_STYLE),
        html.P(id='card_text_3', children=['Sample text.'], style=CARD_TEXT_STYLE)])]), md=4)])

content = html.Div([
    html.H2('Smart Home Energy System Analytics', style=TITLE_STYLE),
    html.Hr(),
    content_fifth_row,
    content_first_row,
    content_fourth_row,
    content_second_row,
    content_third_row],
    style=CONTENT_STYLE)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([sidebar, content])


@app.callback(
    Output('graph_1', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_1(n_clicks, dropdown_value, range_slider_value, slider_value):
    print('\nGraph 1 update.' + '=' * 20, f'\nClicks number: {n_clicks}',
          f'\nRange-slider: {range_slider_value}', f'\nSlider: {slider_value}')
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    df = hard_process_dataframe(df)
    df = resampling_dataframe(df, f'{slider_value}min')
    # Compute the correlation matrix
    corr = df.corr()
    fig = px.imshow(corr)
    return fig


@app.callback(
    Output('graph_2', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_2(n_clicks, dropdown_value, range_slider_value, slider_value):
    print('\nGraph 2 update.' + '=' * 20, f'\nClicks number: {n_clicks}',
          f'\nRange-slider: {range_slider_value}', f'\nSlider: {slider_value}')
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    df = hard_process_dataframe(df)
    df = resampling_dataframe(df, f'{slider_value}min')
    columns = ['Kitchen', 'Living Room', 'Furnace', 'Outside', 'Home Office']
    pie_values = pie_chart(df, columns)
    fig = go.Figure(data=[go.Pie(labels=columns, values=pie_values)])
    return fig


@app.callback(
    Output('graph_3', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_3(n_clicks, dropdown_value, range_slider_value, slider_value):
    print('\nGraph 3 update.' + '=' * 20, f'\nClicks number: {n_clicks}',
          f'\nRange-slider: {range_slider_value}', f'\nSlider: {slider_value}')
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    df = light_process_dataframe(df)
    df = resampling_dataframe(df, f'{slider_value}min')
    columns = ['Kitchen', 'Living Room', 'Furnace', 'Home Office', 'Fridge', 'Microwave', 'Wine Cellar',
               'Dishwasher', 'Well', 'Garage Door', 'Barn']
    pie_values = pie_chart(df, columns)
    fig = go.Figure(data=[go.Pie(labels=columns, values=pie_values)])
    return fig


@app.callback(
    Output('graph_4', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_4(n_clicks, dropdown_value, range_slider_value, slider_value):
    print('\nGraph 4 update.' + '=' * 20, f'\nClicks number: {n_clicks}',
          f'\nRange-slider: {range_slider_value}', f'\nSlider: {slider_value}')
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    df = hard_process_dataframe(df)
    df = resampling_dataframe(df, f'{slider_value}min')
    fig = px.scatter(df, x=dropdown_value[0], y=dropdown_value[1])
    return fig


@app.callback(
    Output('graph_5', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_5(n_clicks, dropdown_value, range_slider_value, slider_value):
    print('\nGraph 5 update.' + '=' * 20, f'\nClicks number: {n_clicks}',
          f'\nRange-slider: {range_slider_value}', f'\nSlider: {slider_value}')
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    df = hard_process_dataframe(df)
    df = resampling_dataframe(df, f'{slider_value}min')
    fig = px.box(df)
    return fig


@app.callback(
    Output('graph_6', 'figure'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_graph_6(n_clicks, dropdown_value, range_slider_value, slider_value):
    print('\nGraph 6 update.' + '=' * 20, f'\nClicks number: {n_clicks}',
          f'\nRange-slider: {range_slider_value}', f'\nSlider: {slider_value}')
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    df = hard_process_dataframe(df)
    df = resampling_dataframe(df, f'{slider_value}min')
    fig = px.line(df)
    return fig


@app.callback(
    Output('card_text_1', 'children'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_card_text_1(n_clicks, dropdown_value, range_slider_value, slider_value):
    return f'{slider_value} minutes'


@app.callback(
    Output('card_text_2', 'children'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_card_text_2(n_clicks, dropdown_value, range_slider_value, slider_value):
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    df = hard_process_dataframe(df)
    df = resampling_dataframe(df, f'{slider_value}min')
    return f'{df.index[0]}'


@app.callback(
    Output('card_text_3', 'children'), [Input('submit_button', 'n_clicks')],
    [State('features', 'value'), State('date_range', 'value'), State('resampling', 'value')])
def update_card_text_3(n_clicks, dropdown_value, range_slider_value, slider_value):
    df = dataframe.iloc[range_slider_value[0]:range_slider_value[1], :]
    df = hard_process_dataframe(df)
    df = resampling_dataframe(df, f'{slider_value}min')
    return f'{df.index[-1]}'


if __name__ == '__main__':
    app.run_server(port=8085)

# ======================================================================================================================
# controls = dbc.FormGroup(
#     [html.P('Dropdown', style={'textAlign': 'center'}),
#      dcc.Dropdown(id='dropdown',
#                   options=[{'label': 'Value One', 'value': 'value1'},
#                            {'label': 'Value Two', 'value': 'value2'},
#                            {'label': 'Value Three', 'value': 'value3'}],
#                   value=['value1'],  # default value
#                   multi=False),
#      html.Br(),
#      html.P('Range Slider', style={'textAlign': 'center'}),
#      dcc.RangeSlider(id='range_slider', min=0, max=20, step=0.5, value=[5, 15]),
#      html.P('Check Box', style={'textAlign': 'center'}),
#      dbc.Card([dbc.Checklist(id='check_list',
#                              options=[{'label': 'Value One', 'value': 'value1'},
#                                       {'label': 'Value Two', 'value': 'value2'},
#                                       {'label': 'Value Three', 'value': 'value3'}],
#                              value=['value1', 'value2'],
#                              inline=True)]),
#      html.Br(),
#      html.P('Radio Items', style={'textAlign': 'center'}),
#      dbc.Card([dbc.RadioItems(id='radio_items',
#                               options=[{'label': 'Value One', 'value': 'value1'},
#                                        {'label': 'Value Two', 'value': 'value2'},
#                                        {'label': 'Value Three', 'value': 'value3'}],
#                               value='value1',
#                               style={'margin': 'auto'})]),
#      html.Br(),
#      dbc.Button(id='submit_button', n_clicks=0, children='Submit', color='primary', block=True)])
