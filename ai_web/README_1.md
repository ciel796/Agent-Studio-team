## Agent Studio Web

此目录提供一个纯 Python 的本地 Web UI，用于展示对话、工具、记忆、模型配置与执行追踪，并通过 HTTP API 对接 `d:\实训-自然语言处理\B方向\agent\code` 的运行时能力。

### 启动

在 `d:\实训-自然语言处理\B方向\agent` 目录下执行：

```powershell
python .\ai_web\server.py
```

浏览器打开：

```
http://127.0.0.1:8010/
```

也可以自定义 host 和端口：

```powershell
python .\ai_web\server.py --host 127.0.0.1 --port 8010
```

### 说明

- UI 已接入 `b1_agent_runtime_1.py`，可直接测试 `adaptive_execute / plan_execute / integrated / react_one_round`
- 当后端返回 `ASK_USER` 时，前端会保留当前会话，下一条消息会自动通过 `resume` 继续同一轮任务
- API 会把每次运行的产物写入 `agent/outputs/web_ui/<conversation_id>/<timestamp>/`
