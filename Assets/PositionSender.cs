// === PositionSender.cs ===
using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class PositionSender : MonoBehaviour
{
    public Rigidbody playerRb;
    public Rigidbody botRb;

    public string serverIP = "127.0.0.1";
    public int serverPort = 5005;

    private TcpClient client;
    private NetworkStream stream;
    private Thread clientThread;
    private bool isConnected = false;

    private Vector3 latestPlayerPos, latestPlayerVel;
    private Quaternion latestPlayerRot;
    private Vector3 latestBotPos, latestBotVel;
    private Quaternion latestBotRot;
    private object positionLock = new object();

    void Start()
    {
        clientThread = new Thread(ConnectAndSendLoop);
        clientThread.IsBackground = true;
        clientThread.Start();
    }

    void Update()
    {
        if (playerRb != null && botRb != null)
        {
            lock (positionLock)
            {
                latestPlayerPos = playerRb.position;
                latestPlayerVel = playerRb.linearVelocity;
                latestPlayerRot = playerRb.rotation;

                latestBotPos = botRb.position;
                latestBotVel = botRb.linearVelocity;
                latestBotRot = botRb.rotation;
            }
        }
    }

    void ConnectAndSendLoop()
    {
        try
        {
            client = new TcpClient(serverIP, serverPort);
            stream = client.GetStream();
            isConnected = true;
            Debug.Log("[TCP] Connected to server");

            while (isConnected)
            {
                Vector3 pp, pv, bp, bv;
                Quaternion pr, br;

                lock (positionLock)
                {
                    pp = latestPlayerPos;
                    pv = latestPlayerVel;
                    pr = latestPlayerRot;
                    bp = latestBotPos;
                    bv = latestBotVel;
                    br = latestBotRot;
                }

                PositionPayload payload = new PositionPayload(pp, pv, pr, bp, bv, br);
                string json = JsonUtility.ToJson(payload) + "\n";
                byte[] data = Encoding.ASCII.GetBytes(json);
                stream.Write(data, 0, data.Length);
                stream.Flush();

                Thread.Sleep(33);  // ~30 FPS
            }
        }
        catch (Exception e)
        {
            Debug.LogError("[TCP] Error: " + e.Message);
            isConnected = false;
        }
        finally
        {
            stream?.Close();
            client?.Close();
        }
    }

    void OnApplicationQuit()
    {
        isConnected = false;
        stream?.Close();
        client?.Close();
        clientThread?.Abort();
    }

    [Serializable]
    public class Vector3Data
    {
        public float x, y, z;
        public Vector3Data(Vector3 vec) { x = vec.x; y = vec.y; z = vec.z; }
    }

    [Serializable]
    public class QuaternionData
    {
        public float x, y, z, w;
        public QuaternionData(Quaternion quat) { x = quat.x; y = quat.y; z = quat.z; w = quat.w; }
    }

    [Serializable]
    public class PositionPayload
    {
        public ObjectData player;
        public ObjectData bot;

        public PositionPayload(Vector3 pp, Vector3 pv, Quaternion pr, Vector3 bp, Vector3 bv, Quaternion br)
        {
            player = new ObjectData(pp, pv, pr);
            bot = new ObjectData(bp, bv, br);
        }
    }

    [Serializable]
    public class ObjectData
    {
        public Vector3Data position;
        public Vector3Data velocity;
        public QuaternionData rotation;

        public ObjectData(Vector3 pos, Vector3 vel, Quaternion rot)
        {
            position = new Vector3Data(pos);
            velocity = new Vector3Data(vel);
            rotation = new QuaternionData(rot);
        }
    }
}
