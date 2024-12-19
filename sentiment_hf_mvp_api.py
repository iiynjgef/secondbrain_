import requests
import time
import json


class SentimentAnalysisAPI:
    def __init__(self, base_url, max_retries=10, wait_time=1):
        """
        初始化 API 客户端。
        :param base_url: Hugging Face Space 的基础 URL。
        :param max_retries: 最大轮询次数。
        :param wait_time: 每次轮询之间的等待时间（秒）。
        """
        self.base_url = base_url
        self.api_url = f"{self.base_url}/gradio_api/call/predict"
        self.max_retries = max_retries
        self.wait_time = wait_time

    def submit_request(self, text):
        """
        提交文本到 API 并获取事件 ID。
        :param text: 要分析的文本。
        :return: 事件 ID。
        :raises: ValueError 如果提交失败或返回无效。
        """
        headers = {"Content-Type": "application/json"}
        data = {"data": [text]}

        response = requests.post(self.api_url, headers=headers, json=data)
        if response.status_code == 200:
            try:
                response_json = response.json()
                event_id = response_json.get("event_id", response_json.get("id"))
                if not event_id:
                    raise ValueError("Response missing 'event_id' or 'id'.")
                return event_id
            except requests.exceptions.JSONDecodeError:
                raise ValueError(f"Response is not valid JSON. Raw response: {response.text}")
        else:
            raise ValueError(f"Failed to submit data: {response.status_code}, {response.text}")

    def get_result(self, event_id):
        """
        使用事件 ID 获取分析结果。
        :param event_id: 提交请求返回的事件 ID。
        :return: 分析结果。
        :raises: TimeoutError 如果超过最大轮询次数。
        :raises: ValueError 如果获取失败。
        """
        result_url = f"{self.base_url}/gradio_api/call/predict/{event_id}"
        retries = 0

        while retries < self.max_retries:
            result = requests.get(result_url)
            if result.status_code == 200:
                try:
                    # 手动解析事件流格式
                    raw_text = result.text.strip()
                    if "data:" in raw_text:
                        data_str = raw_text.split("data:")[1].strip()
                        parsed_data = json.loads(data_str)  # 将字符串解析为 JSON
                        return parsed_data  # 返回解析后的结果
                    else:
                        raise ValueError("No 'data:' found in response.")
                except Exception as e:
                    raise ValueError(f"Error parsing response: {e}")
            elif result.status_code == 404:
                time.sleep(self.wait_time)
                retries += 1
            else:
                raise ValueError(f"Failed to fetch result: {result.status_code}, {result.text}")

        raise TimeoutError("Max retries exceeded while waiting for result.")


# 示例用法
if __name__ == "__main__":
    base_url = "https://jozoe-sentiment-analysis-space2.hf.space"
    api_client = SentimentAnalysisAPI(base_url)

    try:
        # 提交请求并获取结果
        text_to_analyze = "Hello!!"
        print("Submitting request...")
        event_id = api_client.submit_request(text_to_analyze)
        print(f"Event ID: {event_id}")

        print("Fetching result...")
        result = api_client.get_result(event_id)
        print("Sentiment Analysis Result:", result)
    except Exception as e:
        print("Error:", e)
