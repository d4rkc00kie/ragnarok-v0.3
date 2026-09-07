// Ollama TUI ↔ Unity avatar bridge (Linux / Fedora friendly)
// Drop this script on a GameObject in your scene (with your avatar + AudioSource).
//
// Listens on TCP 127.0.0.1:8765 for newline-delimited JSON messages from ollama-tui.
//
// Messages:
//   {"type":"hello","client":"ollama-tui","version":1}
//   {"type":"speak","id":"...","text":"...","audio_path":"/path/file.mp3","duration":3.2}
//   {"type":"stop"}
//   {"type":"idle"}
//   {"type":"ping"}
//
// Requires: NativeFileBrowser optional — UnityWebRequestMultimedia works for file:// on Linux.
// Lip sync: drive Animator float "MouthOpen" or blend shape from AudioSource spectrum.

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Networking;

[Serializable]
public class OllamaTuiMessage
{
    public string type;
    public string id;
    public string text;
    public string audio_path;
    public float duration;
    public string client;
    public int version;
}

public class OllamaTuiAvatarBridge : MonoBehaviour
{
    [Header("Network")]
    public string listenHost = "127.0.0.1";
    public int listenPort = 8765;

    [Header("Avatar")]
    public Animator animator;
    public string talkBoolParam = "IsTalking";
    public string mouthFloatParam = "MouthOpen";
    public AudioSource voiceSource;
    public SkinnedMeshRenderer faceMesh;
    public int mouthBlendShapeIndex = 0;

    [Header("Behaviour")]
    public bool loadAudioFromPath = true;
    public float idleMouth = 0f;

    private TcpListener _listener;
    private Thread _acceptThread;
    private readonly ConcurrentQueue<OllamaTuiMessage> _queue = new ConcurrentQueue<OllamaTuiMessage>();
    private volatile bool _running;
    private Coroutine _speakRoutine;

    void Start()
    {
        if (voiceSource == null)
            voiceSource = gameObject.AddComponent<AudioSource>();
        if (animator == null)
            animator = GetComponentInChildren<Animator>();

        _running = true;
        try
        {
            _listener = new TcpListener(IPAddress.Parse(listenHost), listenPort);
            _listener.Start();
            _acceptThread = new Thread(AcceptLoop) { IsBackground = true };
            _acceptThread.Start();
            Debug.Log($"[OllamaTUI] Listening on {listenHost}:{listenPort}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[OllamaTUI] Failed to bind {listenHost}:{listenPort}: {e.Message}");
        }
    }

    void OnDestroy()
    {
        _running = false;
        try { _listener?.Stop(); } catch { /* ignore */ }
        try { _acceptThread?.Join(500); } catch { /* ignore */ }
    }

    void Update()
    {
        while (_queue.TryDequeue(out var msg))
            HandleMessage(msg);

        // simple spectrum lip-sync while playing
        if (voiceSource != null && voiceSource.isPlaying && !string.IsNullOrEmpty(mouthFloatParam))
        {
            float level = GetLoudness(voiceSource);
            SetMouth(level);
        }
    }

    void HandleMessage(OllamaTuiMessage msg)
    {
        if (msg == null || string.IsNullOrEmpty(msg.type))
            return;

        switch (msg.type)
        {
            case "hello":
                Debug.Log($"[OllamaTUI] Client hello: {msg.client} v{msg.version}");
                break;
            case "ping":
                break;
            case "stop":
                StopSpeaking();
                break;
            case "idle":
                SetTalking(false);
                SetMouth(idleMouth);
                break;
            case "speak":
                if (_speakRoutine != null)
                    StopCoroutine(_speakRoutine);
                _speakRoutine = StartCoroutine(SpeakRoutine(msg));
                break;
            default:
                Debug.LogWarning($"[OllamaTUI] Unknown type: {msg.type}");
                break;
        }
    }

    IEnumerator SpeakRoutine(OllamaTuiMessage msg)
    {
        SetTalking(true);

        bool played = false;
        if (loadAudioFromPath && !string.IsNullOrEmpty(msg.audio_path) && File.Exists(msg.audio_path))
        {
            string url = "file://" + msg.audio_path;
            using (var req = UnityWebRequestMultimedia.GetAudioClip(url, AudioType.UNKNOWN))
            {
                yield return req.SendWebRequest();
#if UNITY_2020_2_OR_NEWER
                if (req.result == UnityWebRequest.Result.Success)
#else
                if (!req.isNetworkError && !req.isHttpError)
#endif
                {
                    var clip = DownloadHandlerAudioClip.GetContent(req);
                    voiceSource.clip = clip;
                    voiceSource.Play();
                    played = true;
                    while (voiceSource.isPlaying)
                        yield return null;
                }
                else
                {
                    Debug.LogWarning($"[OllamaTUI] Audio load failed: {req.error}");
                }
            }
        }

        if (!played)
        {
            float wait = msg.duration > 0.1f ? msg.duration : Mathf.Max(1.5f, (msg.text?.Split(' ').Length ?? 4) / 2.5f);
            float t = 0f;
            while (t < wait)
            {
                // procedural mouth flap
                SetMouth(0.2f + 0.65f * Mathf.Abs(Mathf.Sin(Time.time * 12f)));
                t += Time.deltaTime;
                yield return null;
            }
        }

        SetTalking(false);
        SetMouth(idleMouth);
        _speakRoutine = null;
    }

    void StopSpeaking()
    {
        if (_speakRoutine != null)
        {
            StopCoroutine(_speakRoutine);
            _speakRoutine = null;
        }
        if (voiceSource != null && voiceSource.isPlaying)
            voiceSource.Stop();
        SetTalking(false);
        SetMouth(idleMouth);
    }

    void SetTalking(bool value)
    {
        if (animator != null && !string.IsNullOrEmpty(talkBoolParam))
            animator.SetBool(talkBoolParam, value);
    }

    void SetMouth(float amount)
    {
        amount = Mathf.Clamp01(amount);
        if (animator != null && !string.IsNullOrEmpty(mouthFloatParam))
            animator.SetFloat(mouthFloatParam, amount);
        if (faceMesh != null && mouthBlendShapeIndex >= 0)
            faceMesh.SetBlendShapeWeight(mouthBlendShapeIndex, amount * 100f);
    }

    static float GetLoudness(AudioSource src)
    {
        if (src == null || src.clip == null)
            return 0f;
        float[] samples = new float[256];
        src.GetOutputData(samples, 0);
        float sum = 0f;
        for (int i = 0; i < samples.Length; i++)
            sum += samples[i] * samples[i];
        return Mathf.Clamp01(Mathf.Sqrt(sum / samples.Length) * 8f);
    }

    void AcceptLoop()
    {
        while (_running)
        {
            try
            {
                var client = _listener.AcceptTcpClient();
                var t = new Thread(() => ClientLoop(client)) { IsBackground = true };
                t.Start();
            }
            catch
            {
                if (!_running) break;
            }
        }
    }

    void ClientLoop(TcpClient client)
    {
        using (client)
        using (var stream = client.GetStream())
        using (var reader = new StreamReader(stream, Encoding.UTF8))
        {
            while (_running && client.Connected)
            {
                string line;
                try { line = reader.ReadLine(); }
                catch { break; }
                if (line == null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                try
                {
                    var msg = JsonUtility.FromJson<OllamaTuiMessage>(line);
                    if (msg != null)
                        _queue.Enqueue(msg);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[OllamaTUI] Bad JSON: {e.Message}");
                }
            }
        }
    }
}
