import { PAYLOAD_CONSTANTS } from '../types/payload';
import {
  BackendInferenceResponse,
  BackendInferenceResponseSchema,
  ConnectionState,
} from '../types/response';

export interface WebSocketServiceOptions {
  url?: string;
  heartbeatIntervalMs?: number;
  maxQueueCapacity?: number;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
}

export type MessageCallback = (response: BackendInferenceResponse) => void;
export type ConnectionStateCallback = (state: ConnectionState) => void;
export type LatencyCallback = (latencyMs: number) => void;

/**
 * High-performance Circular Queue holding ArrayBuffer payloads during network dropouts.
 */
export class CircularFrameQueue {
  private buffer: (ArrayBuffer | null)[];
  private head = 0;
  private tail = 0;
  private size = 0;
  private capacity: number;

  constructor(capacity = 50) {
    this.capacity = capacity;
    this.buffer = new Array(capacity).fill(null);
  }

  public enqueue(item: ArrayBuffer): void {
    if (this.size === this.capacity) {
      // Overwrite oldest item if buffer is full
      this.head = (this.head + 1) % this.capacity;
    } else {
      this.size++;
    }

    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.capacity;
  }

  public dequeue(): ArrayBuffer | null {
    if (this.size === 0) return null;

    const item = this.buffer[this.head];
    this.buffer[this.head] = null;
    this.head = (this.head + 1) % this.capacity;
    this.size--;

    return item;
  }

  public clear(): void {
    this.buffer.fill(null);
    this.head = 0;
    this.tail = 0;
    this.size = 0;
  }

  public getSize(): number {
    return this.size;
  }
}

export class WebSocketService {
  private url: string;
  private socket: WebSocket | null = null;
  private connectionState: ConnectionState = 'DISCONNECTED';
  private frameQueue: CircularFrameQueue;
  private heartbeatIntervalMs: number;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  // Exponential backoff parameters
  private initialBackoffMs: number;
  private maxBackoffMs: number;
  private currentBackoffMs: number;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private isManuallyClosed = false;

  // Listeners
  private messageListeners: Set<MessageCallback> = new Set();
  private stateListeners: Set<ConnectionStateCallback> = new Set();
  private latencyListeners: Set<LatencyCallback> = new Set();

  constructor(options: WebSocketServiceOptions = {}) {
    this.url = options.url || 'wss://eval.mentalhealth.ai/evaluate/ws';
    this.heartbeatIntervalMs = options.heartbeatIntervalMs || 30000; // 30s
    this.initialBackoffMs = options.initialBackoffMs || 1000; // 1s
    this.maxBackoffMs = options.maxBackoffMs || 30000; // 30s
    this.currentBackoffMs = this.initialBackoffMs;
    this.frameQueue = new CircularFrameQueue(options.maxQueueCapacity || 50);
  }

  public connect(customUrl?: string): void {
    if (customUrl) this.url = customUrl;
    if (this.socket && (this.socket.readyState === WebSocket.CONNECTING || this.socket.readyState === WebSocket.OPEN)) {
      return;
    }

    this.isManuallyClosed = false;
    this.setConnectionState(this.reconnectAttempts > 0 ? 'RECONNECTING' : 'CONNECTING');

    try {
      this.socket = new WebSocket(this.url);
      this.socket.binaryType = 'arraybuffer';

      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onerror = this.handleError.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
    } catch (err) {
      this.scheduleReconnect();
    }
  }

  public disconnect(): void {
    this.isManuallyClosed = true;
    this.clearTimers();

    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }

    this.frameQueue.clear();
    this.setConnectionState('DISCONNECTED');
  }

  /**
   * Sends a 1,632-byte binary ArrayBuffer over the WebSocket connection.
   * If disconnected or reconnecting, queues frame in the 50-item circular queue.
   */
  public sendBinaryPayload(buffer: ArrayBuffer): boolean {
    if (buffer.byteLength !== PAYLOAD_CONSTANTS.TOTAL_BUFFER_BYTE_SIZE) {
      console.warn(
        `[WebSocketService] Buffer rejected: expected ${PAYLOAD_CONSTANTS.TOTAL_BUFFER_BYTE_SIZE} bytes, received ${buffer.byteLength}`
      );
      return false;
    }

    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      const sendStartTime = performance.now();
      try {
        this.socket.send(buffer);
        // Track client egress timing
        const transitEst = performance.now() - sendStartTime;
        this.notifyLatency(Math.max(5, transitEst));
        return true;
      } catch (err) {
        this.frameQueue.enqueue(buffer);
        return false;
      }
    } else {
      // Queue frame during network dropouts
      this.frameQueue.enqueue(buffer);
      return false;
    }
  }

  private handleOpen(): void {
    this.setConnectionState('CONNECTED');
    this.reconnectAttempts = 0;
    this.currentBackoffMs = this.initialBackoffMs;

    this.startHeartbeat();
    this.flushQueuedFrames();
  }

  private handleMessage(event: MessageEvent): void {
    this.resetHeartbeat();

    try {
      let rawData: any;
      if (typeof event.data === 'string') {
        rawData = JSON.parse(event.data);
      } else if (event.data instanceof ArrayBuffer) {
        const text = new TextDecoder().decode(event.data);
        rawData = JSON.parse(text);
      }

      if (rawData) {
        if (rawData.type === 'PONG') return; // Heartbeat pong response

        const parsed = BackendInferenceResponseSchema.safeParse(rawData);
        if (parsed.success) {
          this.notifyMessage(parsed.data);
          if (parsed.data.transitLatencyMs) {
            this.notifyLatency(parsed.data.transitLatencyMs);
          }
        }
      }
    } catch {
      // Message parsing error handling
    }
  }

  private handleError(): void {
    if (this.connectionState === 'CONNECTED') {
      this.setConnectionState('RECONNECTING');
    }
  }

  private handleClose(): void {
    this.clearTimers();
    if (!this.isManuallyClosed) {
      this.setConnectionState('RECONNECTING');
      this.scheduleReconnect();
    } else {
      this.setConnectionState('DISCONNECTED');
    }
  }

  private flushQueuedFrames(): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;

    let queuedFrame = this.frameQueue.dequeue();
    let count = 0;

    // Rate-limited burst flush of queued frames
    while (queuedFrame && count < 50) {
      try {
        this.socket.send(queuedFrame);
      } catch {
        break;
      }
      count++;
      queuedFrame = this.frameQueue.dequeue();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer || this.isManuallyClosed) return;

    this.reconnectAttempts++;
    // Exponential backoff with 20% jitter
    const jitter = (Math.random() * 0.4 - 0.2) * this.currentBackoffMs;
    const delay = Math.min(this.maxBackoffMs, this.currentBackoffMs + jitter);

    this.currentBackoffMs = Math.min(this.maxBackoffMs, this.currentBackoffMs * 2.0);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.clearHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify({ type: 'PING', timestamp: Date.now() }));
      }
    }, this.heartbeatIntervalMs);
  }

  private resetHeartbeat(): void {
    if (this.heartbeatTimer) {
      this.startHeartbeat();
    }
  }

  private clearHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private clearTimers(): void {
    this.clearHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private setConnectionState(state: ConnectionState): void {
    if (this.connectionState !== state) {
      this.connectionState = state;
      this.stateListeners.forEach((cb) => cb(state));
    }
  }

  private notifyMessage(data: BackendInferenceResponse): void {
    this.messageListeners.forEach((cb) => cb(data));
  }

  private notifyLatency(latencyMs: number): void {
    this.latencyListeners.forEach((cb) => cb(latencyMs));
  }

  // Event Listener Subscriptions
  public onMessage(cb: MessageCallback): () => void {
    this.messageListeners.add(cb);
    return () => this.messageListeners.delete(cb);
  }

  public onConnectionState(cb: ConnectionStateCallback): () => void {
    this.stateListeners.add(cb);
    return () => this.stateListeners.delete(cb);
  }

  public onLatency(cb: LatencyCallback): () => void {
    this.latencyListeners.add(cb);
    return () => this.latencyListeners.delete(cb);
  }

  public getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  public getQueueSize(): number {
    return this.frameQueue.getSize();
  }
}
