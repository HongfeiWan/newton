type BridgeState = 'idle' | 'connecting' | 'listening' | 'error';

interface QuestVoiceCommandBridgeOptions {
  buttonId: string;
  statusId: string;
  uplinkSampleRate?: number;
}

export class QuestVoiceCommandBridge {
  private readonly buttonId: string;
  private readonly statusId: string;
  private readonly uplinkSampleRate: number;
  private button: HTMLButtonElement | null = null;
  private statusEl: HTMLElement | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private processorNode: ScriptProcessorNode | null = null;
  private muteGainNode: GainNode | null = null;
  private socket: WebSocket | null = null;
  private state: BridgeState = 'idle';
  private boundToggle = () => {
    void this.toggle();
  };

  constructor(options: QuestVoiceCommandBridgeOptions) {
    this.buttonId = options.buttonId;
    this.statusId = options.statusId;
    this.uplinkSampleRate = options.uplinkSampleRate ?? 16000;
  }

  initialize(): void {
    this.button = document.getElementById(this.buttonId) as HTMLButtonElement | null;
    this.statusEl = document.getElementById(this.statusId);
    if (!this.button || !this.statusEl) {
      return;
    }

    this.button.addEventListener('click', this.boundToggle);
    this.render();
  }

  cleanup(): void {
    if (this.button) {
      this.button.removeEventListener('click', this.boundToggle);
    }
    void this.stop();
  }

  private async toggle(): Promise<void> {
    if (this.state === 'connecting' || this.state === 'listening') {
      await this.stop();
      return;
    }
    await this.start();
  }

  private async start(): Promise<void> {
    if (!this.button || !this.statusEl) {
      return;
    }

    this.state = 'connecting';
    this.render('Requesting microphone permission…');

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const socket = await this.openSocket();
      const audioContext = new AudioContext();
      const sourceNode = audioContext.createMediaStreamSource(mediaStream);
      const processorNode = audioContext.createScriptProcessor(4096, 1, 1);
      const muteGainNode = audioContext.createGain();
      muteGainNode.gain.value = 0.0;

      processorNode.onaudioprocess = (event: AudioProcessingEvent) => {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
          return;
        }
        const input = event.inputBuffer.getChannelData(0);
        const pcm16 = downsampleFloat32ToPcm16(
          input,
          event.inputBuffer.sampleRate,
          this.uplinkSampleRate
        );
        if (pcm16.byteLength > 0) {
          this.socket.send(pcm16.buffer);
        }
      };

      sourceNode.connect(processorNode);
      processorNode.connect(muteGainNode);
      muteGainNode.connect(audioContext.destination);

      this.mediaStream = mediaStream;
      this.socket = socket;
      this.audioContext = audioContext;
      this.sourceNode = sourceNode;
      this.processorNode = processorNode;
      this.muteGainNode = muteGainNode;
      this.state = 'listening';
      this.socket.send(
        JSON.stringify({
          type: 'session_start',
          sampleRate: this.uplinkSampleRate,
        })
      );
      this.render('Quest mic is streaming to host ASR');
    } catch (error) {
      console.error('[quest-voice-bridge] failed to start:', error);
      this.state = 'error';
      this.render(
        `Voice bridge failed: ${error instanceof Error ? error.message : String(error)}`
      );
      await this.stop(false);
    }
  }

  private async stop(resetState: boolean = true): Promise<void> {
    if (this.processorNode) {
      this.processorNode.onaudioprocess = null;
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.muteGainNode) {
      this.muteGainNode.disconnect();
      this.muteGainNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
    if (this.socket) {
      try {
        if (this.socket.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify({ type: 'session_stop' }));
        }
      } catch (_) {}
      this.socket.close();
      this.socket = null;
    }

    if (resetState) {
      this.state = 'idle';
      this.render('Quest microphone uplink is off');
    }
  }

  private openSocket(): Promise<WebSocket> {
    const url = this.resolveSocketUrl();
    console.info('[quest-voice-bridge] connecting to', url);
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(url);
      socket.binaryType = 'arraybuffer';

      const cleanup = () => {
        socket.removeEventListener('open', onOpen);
        socket.removeEventListener('error', onError);
      };
      const onOpen = () => {
        cleanup();
        console.info('[quest-voice-bridge] connected');
        resolve(socket);
      };
      const onError = (event: Event) => {
        cleanup();
        console.error('[quest-voice-bridge] websocket error', event);
        reject(new Error(`Could not connect to ${url}`));
      };

      socket.addEventListener('open', onOpen);
      socket.addEventListener('error', onError);
      socket.addEventListener('close', event => {
        console.warn(
          '[quest-voice-bridge] disconnected',
          `code=${event.code}`,
          `reason=${event.reason || '(empty)'}`
        );
        if (this.state === 'listening') {
          this.state = 'idle';
          this.render('Quest microphone uplink disconnected');
        }
      });
    });
  }

  private resolveSocketUrl(): string {
    const params = new URLSearchParams(window.location.search);
    const urlParam = params.get('questVoiceUrl');
    const portParam = params.get('questVoicePort');
    const pathParam = params.get('questVoicePath');
    const storageUrlKey = 'teleop.questVoiceUrl';
    const storagePortKey = 'teleop.questVoicePort';
    const storagePathKey = 'teleop.questVoicePath';
    const proxyUrl = `${window.location.origin.replace(/^http/, 'ws')}/quest-voice`;

    if (urlParam) {
      window.localStorage.setItem(storageUrlKey, urlParam);
      window.localStorage.removeItem(storagePortKey);
      window.localStorage.removeItem(storagePathKey);
      return urlParam;
    }

    if (pathParam) {
      const normalizedPath = pathParam.startsWith('/') ? pathParam : `/${pathParam}`;
      window.localStorage.setItem(storagePathKey, normalizedPath);
      window.localStorage.removeItem(storageUrlKey);
      window.localStorage.removeItem(storagePortKey);
      return `${window.location.origin.replace(/^http/, 'ws')}${normalizedPath}`;
    }

    if (portParam) {
      if (portParam === 'proxy' || portParam === 'default') {
        window.localStorage.removeItem(storageUrlKey);
        window.localStorage.removeItem(storagePortKey);
        window.localStorage.removeItem(storagePathKey);
        return proxyUrl;
      }
      window.localStorage.setItem(storagePortKey, portParam);
      window.localStorage.removeItem(storageUrlKey);
      window.localStorage.removeItem(storagePathKey);
      return `${window.location.protocol.replace(/^http/, 'ws')}//${window.location.hostname}:${portParam}`;
    }

    const storedUrl = window.localStorage.getItem(storageUrlKey);
    if (storedUrl) {
      return storedUrl;
    }

    const storedPath = window.localStorage.getItem(storagePathKey);
    if (storedPath) {
      return `${window.location.origin.replace(/^http/, 'ws')}${storedPath}`;
    }

    const storedPort = window.localStorage.getItem(storagePortKey);
    if (storedPort) {
      return `${window.location.protocol.replace(/^http/, 'ws')}//${window.location.hostname}:${storedPort}`;
    }

    return proxyUrl;
  }

  private render(statusOverride?: string): void {
    if (!this.button || !this.statusEl) {
      return;
    }

    if (this.state === 'connecting') {
      this.button.textContent = 'Stop Voice';
      this.button.disabled = true;
    } else if (this.state === 'listening') {
      this.button.textContent = 'Stop Voice';
      this.button.disabled = false;
    } else {
      this.button.textContent = 'Enable Voice';
      this.button.disabled = false;
    }

    const status =
      statusOverride ??
      (this.state === 'listening'
        ? 'Quest microphone uplink is active'
        : this.state === 'error'
          ? 'Quest microphone uplink failed'
          : 'Quest microphone uplink is off');
    this.statusEl.textContent = status;
  }
}

function downsampleFloat32ToPcm16(
  input: Float32Array,
  inputSampleRate: number,
  targetSampleRate: number
): Int16Array {
  if (input.length === 0) {
    return new Int16Array(0);
  }

  const ratio = inputSampleRate / targetSampleRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Int16Array(outputLength);

  let outputIndex = 0;
  let inputIndex = 0;
  while (outputIndex < outputLength) {
    const nextInputIndex = Math.min(input.length, Math.round((outputIndex + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (; inputIndex < nextInputIndex; inputIndex += 1) {
      sum += input[inputIndex];
      count += 1;
    }
    const sample = count > 0 ? sum / count : input[Math.min(input.length - 1, inputIndex)];
    const clamped = Math.max(-1, Math.min(1, sample));
    output[outputIndex] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    outputIndex += 1;
  }

  return output;
}
