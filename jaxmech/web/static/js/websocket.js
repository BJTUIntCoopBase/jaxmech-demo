/**
 * jaxmech Web Frontend — WebSocket connection manager
 */
const WS = {
  connect(taskId, { onMessage, onClose }) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/tasks/${taskId}/logs`;
    const ws = new WebSocket(url);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    ws.onclose = () => { if (onClose) onClose(); };
    ws.onerror = (err) => { console.error('WS error:', err); };

    return ws;
  }
};
