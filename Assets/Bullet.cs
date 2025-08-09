using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System;

[RequireComponent(typeof(Rigidbody), typeof(Collider))]
public class Bullet : MonoBehaviour
{
    public float speed = 50f;
    public float lifeTime = 5f;
    public string ownerTag; // "Player" or "Bot"
    public string shooterId;
    public string serverIP = "127.0.0.1";
    public int serverPort = 5000;

    private Rigidbody rb;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = false;
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
    }

    void Start()
    {
        rb.linearVelocity = transform.forward * speed;
        Destroy(gameObject, lifeTime);
    }

    void OnTriggerEnter(Collider other)
    {
        if (!string.IsNullOrEmpty(ownerTag) && other.CompareTag(ownerTag))
            return;

        if (other.CompareTag("Player") || other.CompareTag("Bot"))
        {
            bool hitPlayer = other.CompareTag("Player");

            if (ownerTag == "Bot")
                SendHitReport(hitPlayer);

            Destroy(other.gameObject); // optional
        }

        Destroy(gameObject);
    }

    private void SendHitReport(bool positive)
    {
        try
        {
            using (TcpClient client = new TcpClient())
            {
                client.Connect(serverIP, serverPort);
                NetworkStream st = client.GetStream();

                var msg = new SimpleMsg(shooterId, positive);
                string json = JsonUtility.ToJson(msg) + "\n";
                byte[] data = Encoding.UTF8.GetBytes(json);
                st.Write(data, 0, data.Length);
                st.Flush();
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("[Bullet] TCP send failed: " + e.Message);
        }
    }

    [Serializable]
    private class SimpleMsg
    {
        public string agent_id;
        public bool hit;
        public SimpleMsg(string id, bool h) { agent_id = id; hit = h; }
    }
}
