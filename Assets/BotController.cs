using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System;
using Newtonsoft.Json.Linq;

public class BotController : MonoBehaviour
{
    public Rigidbody botRb;
    public float moveSpeed = 10f;
    public float rotationLerpSpeed = 3f;

    private TcpClient client;
    private NetworkStream stream;
    private Thread receiveThread;
    private bool running = true;

    private Quaternion targetRotation = Quaternion.identity;

    void Start()
    {
        receiveThread = new Thread(ReceiveDataLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void FixedUpdate()
    {
        botRb.MoveRotation(Quaternion.Lerp(botRb.rotation, targetRotation, Time.fixedDeltaTime * rotationLerpSpeed));
        botRb.MovePosition(botRb.position + botRb.transform.forward * moveSpeed * Time.fixedDeltaTime);
    }

    void ReceiveDataLoop()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 5006);
            stream = client.GetStream();

            byte[] buffer = new byte[2048];

            while (running)
            {
                int bytesRead = stream.Read(buffer, 0, buffer.Length);
                if (bytesRead == 0) continue;

                string json = Encoding.ASCII.GetString(buffer, 0, bytesRead).Trim();
                JObject parsed = JObject.Parse(json);

                Quaternion predictedRotation = new Quaternion(
                    (float)parsed["rotation"]["x"],
                    (float)parsed["rotation"]["y"],
                    (float)parsed["rotation"]["z"],
                    (float)parsed["rotation"]["w"]
                );

                UnityMainThreadDispatcher.Instance().Enqueue(() =>
                {
                    targetRotation = predictedRotation;
                    Debug.Log("[BOT] Received target rotation: " + targetRotation);
                });
            }
        }
        catch (Exception e)
        {
            Debug.LogError("[BotController] Error: " + e.Message);
        }
    }

    void OnApplicationQuit()
    {
        running = false;
        stream?.Close();
        client?.Close();
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort();
    }
}
