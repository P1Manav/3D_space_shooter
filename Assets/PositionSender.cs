// PositionSender.cs
using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;

public class PositionSender : MonoBehaviour
{
    public Transform player;
    public Transform botTransform;
    public BotController botController; // assign in inspector
    public string agentId = "bot_1";
    public string serverIP = "127.0.0.1";
    public int serverPort = 5000;
    public float sendInterval = 0.05f; // seconds

    private Thread netThread;
    private volatile bool running = false;
    private TcpClient client;
    private NetworkStream stream;
    private object lockObj = new object();

    private string pendingStateJson = null;
    private Queue<string> incomingResponses = new Queue<string>();
    private string recvBuffer = "";
    private float lastSendTime = 0f;

    void Start()
    {
        if (player == null || botTransform == null || botController == null)
        {
            Debug.LogError("[PositionSender] Assign player, botTransform, botController in inspector!");
            enabled = false;
            return;
        }

        running = true;
        netThread = new Thread(NetLoop) { IsBackground = true };
        netThread.Start();
    }

    void Update()
    {
        if (Time.time - lastSendTime >= sendInterval)
        {
            lastSendTime = Time.time;

            var sd = new StateData()
            {
                agent_id = agentId,
                player_pos = new float[] { player.position.x, player.position.y, player.position.z },
                player_vel = TryGetVelocity(player),
                player_rot = new float[] { player.eulerAngles.x, player.eulerAngles.y, player.eulerAngles.z },

                bot_pos = new float[] { botTransform.position.x, botTransform.position.y, botTransform.position.z },
                bot_vel = TryGetVelocity(botTransform),
                bot_rot = new float[] { botTransform.eulerAngles.x, botTransform.eulerAngles.y, botTransform.eulerAngles.z }
            };

            string json = JsonUtility.ToJson(sd) + "\n";
            lock (lockObj) { pendingStateJson = json; }
        }

        // Process responses on main thread (safe to call Unity APIs here)
        lock (lockObj)
        {
            while (incomingResponses.Count > 0)
            {
                string line = incomingResponses.Dequeue();
                try
                {
                    Prediction p = JsonUtility.FromJson<Prediction>(line);
                    Debug.LogFormat("[PositionSender] Prediction received: yaw={0}, pitch={1}, roll={2}, shoot={3}",
                        p.yaw_delta, p.pitch_delta, p.roll_delta, p.shoot);
                    // Call SetPrediction on main thread (we are already on main thread here)
                    botController.SetPrediction(p.yaw_delta, p.pitch_delta, p.roll_delta, p.shoot);
                }
                catch (Exception e)
                {
                    Debug.LogWarning("[PositionSender] parse fail: " + e.Message + " raw:" + line);
                }
            }
        }
    }

    void OnApplicationQuit()
    {
        running = false;
        try { stream?.Close(); client?.Close(); } catch { }
    }

    private void NetLoop()
    {
        while (running)
        {
            try
            {
                client = new TcpClient();
                client.NoDelay = true;
                client.Connect(serverIP, serverPort);
                stream = client.GetStream();
                stream.ReadTimeout = 2000;
                Debug.Log("[PositionSender] Connected to server " + serverIP + ":" + serverPort);
                break;
            }
            catch (Exception e)
            {
                Debug.LogWarning("[PositionSender] Connection failed: " + e.Message);
                Thread.Sleep(1000);
            }
        }

        while (running)
        {
            try
            {
                string toSend = null;
                lock (lockObj) { toSend = pendingStateJson; pendingStateJson = null; }
                if (!string.IsNullOrEmpty(toSend) && stream != null)
                {
                    byte[] data = Encoding.UTF8.GetBytes(toSend);
                    stream.Write(data, 0, data.Length);
                    stream.Flush();
                }

                // read server replies
                while (client != null && client.Available > 0)
                {
                    byte[] buf = new byte[8192];
                    int read = stream.Read(buf, 0, buf.Length);
                    if (read <= 0) break;
                    string s = Encoding.UTF8.GetString(buf, 0, read);
                    recvBuffer += s;
                    while (recvBuffer.Contains("\n"))
                    {
                        int idx = recvBuffer.IndexOf("\n");
                        string line = recvBuffer.Substring(0, idx).Trim();
                        recvBuffer = recvBuffer.Substring(idx + 1);
                        if (!string.IsNullOrEmpty(line))
                        {
                            lock (lockObj) incomingResponses.Enqueue(line);
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("[PositionSender] NetLoop exception: " + e.Message);
                Reconnect();
            }
            Thread.Sleep(1);
        }
    }

    private void Reconnect()
    {
        try { stream?.Close(); client?.Close(); } catch { }
        Thread.Sleep(500);
        while (running)
        {
            try
            {
                client = new TcpClient { NoDelay = true };
                client.Connect(serverIP, serverPort);
                stream = client.GetStream();
                stream.ReadTimeout = 2000;
                Debug.Log("[PositionSender] Reconnected");
                break;
            }
            catch { Thread.Sleep(1000); }
        }
    }

    private float[] TryGetVelocity(Transform t)
    {
        var rb = t.GetComponent<Rigidbody>();
        if (rb != null) return new float[] { rb.linearVelocity.x, rb.linearVelocity.y, rb.linearVelocity.z };
        return new float[] { 0f, 0f, 0f };
    }

    [Serializable]
    public class StateData
    {
        public string agent_id;
        public float[] player_pos;
        public float[] player_vel;
        public float[] player_rot;
        public float[] bot_pos;
        public float[] bot_vel;
        public float[] bot_rot;
    }

    [Serializable]
    public class Prediction
    {
        public float yaw_delta;
        public float pitch_delta;
        public float roll_delta;
        public bool shoot;
    }
}
