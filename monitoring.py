import inspect
from jinja2 import Template
import jinja2schema
from scipy import stats as st
import pandas as pd
import numpy as np

# support functions
def plt_chart_to_bytes():
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_in_bytes = buf.read()
    buf.close()
    return img_in_bytes
def post_message_to_slack(text, slack_token, channel_or_user_id, slack_user_name='alarm_bot', 
                          slack_icon_url = 'https://cdn-images-1.medium.com/max/1200/1*a1O3xhOq8KWSibZF6Ze5xQ.png', 
                          as_user=False, blocks=None, test_mode = False):
    '''
    ### blocks example
    blocks = [{  
      "type": "section",
      "text": {  
        "type": "mrkdwn",
        "text": "*The script* has run\n successfully on the dev."
      }
    }]
    '''
    import requests
    import json
    
    body_dict = {
        'token': slack_token,
        'channel': channel_or_user_id,
        'text': text,
        'username': slack_user_name,
        'blocks': json.dumps(blocks) if blocks else None,
        'icon_url':slack_icon_url
    }
    if as_user:
        body_dict['as_user'] = 'true'
    send_info = requests.post('https://slack.com/api/chat.postMessage', body_dict).json()
    if test_mode: print(send_info)
def post_file_to_slack(slack_token, channel_or_user_id, text, file_name, file_bytes, file_type=None, title=None, test_mode = False):
    import requests
    import json
    
    send_info = requests.post(
      'https://slack.com/api/files.upload', 
      {
        'token': slack_token,
        'filename': file_name,
        'channels': channel_or_user_id,
        'filetype': file_type,
        'initial_comment':  text,
        'title': title
      },
      files = { 'file': file_bytes }).json()
    if test_mode: print(send_info)
# base classes
class BaseConnection:
    connection_type: 'only_read'
    required_methods = dict(only_read = ['select'], admin = ['select', 'execute', 'create_table', 'create_table_by_select', 'drop_table', 'insert_df_to_db', 'insert_select_to_db', 'table_isin_db'])
    connection_types = required_methods.keys()
    def __init__(self):
        if self.connection_type not in self.connection_types: raise ValueError(f'connection_type property of the class object must be one of the values {self.connection_types}!')
        for method in self.required_methods[self.connection_type]:
            getattr(self, method)  
class BaseQuery:
    query_name: str = None
    q: str = None
    query_args_schema: list = list()
    def __init__(self):
        self.query_args_schema = jinja2schema.infer(self.q)
    def render(self, query_args: dict = dict()):
        return Template(self.q).render(**query_args)
class BaseMetric:
    query = BaseQuery
    connection = BaseConnection
    query_args = dict()
    query_params = dict()
    
    metric_field: str = None
    datetime_field: str = None
    std_count_to_outlier: int = 3
    std_count_ci_list = [3,2,1]
    ci_color_list = ['red', '#2b6ca3', 'green']
    seasonality_fields: list = list()
    window_size_days = None
    metric_name: str = None
    metric_chart_link: str = None
    
    def __init__(self):
        self.query_args = jinja2schema.infer(self.query.q)
        self.metric_name = self.metric_name if self.metric_name else self.metric_field
        if len(self.std_count_ci_list) != len(self.ci_color_list): raise ValueError('std_count_ci_list and ci_color_list must be the same length!')
    def get_data(self, query_params: dict = dict()):
        if not query_params: query_params = self.query_params
        q = Template(self.query.q).render(**query_params)
        data = self.connection().select(q)[[self.datetime_field] + self.seasonality_fields + [self.metric_field]]
        # preprocessing data
        data[self.datetime_field] = pd.to_datetime(data[self.datetime_field])
        for field in self.seasonality_fields:
            data[field] = data[field].astype('str')
        return data.sort_values(by = self.datetime_field)
    def get_eval_data(self, query_params: dict = dict()):
        data = self.get_data(query_params)[:-1] # <- убираем последний период из рассмотрения, т.к. он всегда будет не полным
        observation_start = data[self.datetime_field].min() + pd.Timedelta(f'{self.window_size_days} days') # выделим индексы того куска данных, по которым есть window_size_days число дней
        for ind in data[data[self.datetime_field]>=observation_start].index[::-1]: # далее для каждой строки данных считаем норму и отклонение от нее
            row = data.loc[ind]

            period_start, period_end = row[self.datetime_field] - pd.Timedelta(f'{self.window_size_days} days'), row[self.datetime_field]
            sample = data[(data[self.datetime_field]>=period_start)&(data[self.datetime_field]<period_end)]

            mean = sample[self.metric_field].mean() # считаем среднее по всей выборке без учета сезонности
            std = sample[self.metric_field].std() # аналогично считаем и стандартное отклонение

            seasonality_data = pd.DataFrame() # далее считаем коэффициенты сезонности
            for field in self.seasonality_fields:
                seasonality_field_data = sample.groupby(field).agg(
                    seasonality_mean = (self.metric_field, 'mean'),
                    seasonality_std = (self.metric_field, 'std')
                ).reset_index().rename(columns = {field:'value'}).assign(seasonality_field = field)
                seasonality_field_data = seasonality_field_data[seasonality_field_data['value'] == row[field]]
                seasonality_data = pd.concat([seasonality_data, seasonality_field_data])

            seasonality_data = seasonality_data.assign(sample_mean = mean, sample_std = std)
            seasonality_data = seasonality_data.eval('mean_coef = seasonality_mean / sample_mean\nstd_coef = seasonality_std / sample_std')

            mean_coef = seasonality_data['mean_coef'].product() # коэф. сезонности для среднего
            std_coef = seasonality_data['std_coef'].product() # коэф. сезонности для стандартного отклонения

            mean_norm = mean * mean_coef # нахожу среднее значение с учетом сезонности
            std_norm = std * std_coef # аналогично стандартное отклонение с учётом сезонности
            
            deviation = abs(row[self.metric_field] - mean_norm) / std_norm # тут считаю на сколько стандартных отклонений наше значение отклоняется от среднего с учетом сезонности
            distr = st.norm(0, 1)
            p_value = (1 - distr.cdf(deviation)) * 2 # считаем pvalue
            # сохраняем полученные результаты
            data.loc[ind, 'mean'] = mean_norm
            data.loc[ind, 'std'] = std_norm
            data.loc[ind, 'p_value'] = p_value
            data.loc[ind, 'outlier'] = row[self.metric_field] < mean_norm - self.std_count_to_outlier * std_norm or row[self.metric_field] > mean_norm + self.std_count_to_outlier * std_norm
            data.loc[ind, 'outlier_down'] = row[self.metric_field] < mean_norm - self.std_count_to_outlier * std_norm
            data.loc[ind, 'outlier_up'] = row[self.metric_field] > mean_norm + self.std_count_to_outlier * std_norm
        return data[data[self.datetime_field]>=observation_start]
    def get_chart(self, eval_data = None, query_params = None, return_img = True):
        data = eval_data if eval_data is not None else self.get_eval_data(query_params)
        plt.figure(figsize=(40,7))
        sns.scatterplot(data = data.query('outlier==True'), x = self.datetime_field, y = self.metric_field, color = 'red', label = 'outlier')
        sns.scatterplot(data = data.query('outlier==False'), x = self.datetime_field, y = self.metric_field, color = 'grey', label = 'not outlier', alpha = 0.9)
        for std_count, color in zip(self.std_count_ci_list, self.ci_color_list):
            plt.fill_between(data[self.datetime_field], data['mean'] - std_count * data['std'], data['mean'] + std_count * data['std'], color = color, alpha = 0.15, label = f'confidence interval {std_count} std')
        plt.title(f'Значения метрики {self.metric_name}')
        plt.ylabel(f'{self.metric_name}')
        plt.legend()
        img = plt_chart_to_bytes()
        plt.show()
        if return_img: return img
class BaseMonitoring:
    metric: BaseMetric
    outliers_direction_to_alerting: str = None
    last_outliers_to_alerting: int = None
    metric_window_size_days: int = None
    slack_channel_for_alerting_messages: str = None
    eval_data: pd.core.frame.DataFrame = None
    def __init__(self):
        self.metric().window_size_days = self.metric_window_size_days if self.metric_window_size_days else self.metric_window_size_days
        if self.outliers_direction_to_alerting not in ['one','both','down','up']: raise ValueError('''outliers_direction_to_alerting must be value from list ['one','both','down','up']!''')
    def check_metric(self, query_params: dict = dict()):
        eval_data = self.metric().get_eval_data(query_params)
        self.eval_data = eval_data
        alert = False
        if self.outliers_direction_to_alerting == 'one':
            alert = np.count_nonzero(eval_data[-self.last_outliers_to_alerting:]['outlier_down']) == self.last_outliers_to_alerting or np.count_nonzero(eval_data[-self.last_outliers_to_alerting:]['outlier_up']) == self.last_outliers_to_alerting
        elif self.outliers_direction_to_alerting == 'both':
            alert = np.count_nonzero(eval_data[-self.last_outliers_to_alerting:]['outlier']) == self.last_outliers_to_alerting
        elif self.outliers_direction_to_alerting == 'down':
            alert = np.count_nonzero(eval_data[-self.last_outliers_to_alerting:]['outlier_down']) == self.last_outliers_to_alerting
        elif self.outliers_direction_to_alerting == 'up':
            alert = np.count_nonzero(eval_data[-self.last_outliers_to_alerting:]['outlier_up']) == self.last_outliers_to_alerting
        return alert
    def get_chart(self):
        return self.metric().get_chart(eval_data = self.eval_data, return_img = True)
    def get_alerting_message(self):
        metric_name = self.metric().metric_name
        metric_chart_link = self.metric().metric_chart_link
        
        alerting_message = f''':exclamation: *{self.last_outliers_to_alerting} раза подряд аномальные значения метрики '{metric_name}'*\n''' # начало сообщения
        link_info = f'Ссылка на дэш: {metric_chart_link}' # ссылка на деш, указанная в свойсвтах метрики
        outliers_messages = [] # далее формируем для каждого отклонения свое сообщение
        for ind in self.eval_data[-self.last_outliers_to_alerting:].index:
            row = self.eval_data.loc[ind]
            metric_datetime = row[self.metric.datetime_field]
            metric_value = row[self.metric.metric_field]
            metric_pvalue = row['p_value']
            outlier_message = '''*Время:* _{}_\n*Значение метрики {}*: _{:.2f}_\n*P-value*: _{:.4f}_'''\
            .format(metric_datetime, metric_name, metric_value, metric_pvalue)
            outliers_messages.append(outlier_message)
        # собираем по частям будущий алерт
        blocks = [ 
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": alerting_message + link_info}
            },
        ] 
        for outlier_message in outliers_messages:
            blocks.append({"type": "divider"})
            blocks.append({"type": "section",
             "text": {  
               "type": "mrkdwn",
               "text": outlier_message}})
        blocks.append({"type": "divider"})
        blocks.append({"type": "section",
         "text": {  
           "type": "mrkdwn",
           "text": '_*** P-value - вероятность получить такое или еще более отклоняющееся от среднего знач-е при условии того, что метрика ведет себя обычным образом в рамках погрешности._'}})
        return blocks
    def send_message(self, message_blocks, metric_chart_in_bytes, slack_token, test_mode = False):
        post_message_to_slack('', slack_token, self.slack_channel_for_alerting_messages, as_user=True, blocks=message_blocks, test_mode = test_mode)
        post_file_to_slack(slack_token, self.slack_channel_for_alerting_messages, '', f'{self.metric.metric_name}.png', metric_chart_in_bytes, test_mode = test_mode)  
    def run(self, slack_token, query_params: dict = dict(), test_mode = False):
        info_string = f'metric_name: {self.metric().metric_name}\
        \noutliers_direction_to_alerting: {self.outliers_direction_to_alerting}\
        \nlast_outliers_to_alerting: {self.last_outliers_to_alerting}\
        \nmetric_window_size_days: {self.metric_window_size_days}\
        \nslack_channel_for_alerting_messages: {self.slack_channel_for_alerting_messages}'
        alert = self.check_metric(query_params)
        if test_mode: alert = True; print(info_string)
        if alert:
            message_blocks = self.get_alerting_message()
            metric_chart_in_bytes = self.get_chart()
            send_info = self.send_message(message_blocks, metric_chart_in_bytes, slack_token = slack_token, test_mode = test_mode)
            if test_mode: print(send_info)