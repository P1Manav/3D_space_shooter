using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class PlayerMovement : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;
    private string serverIP = "127.0.0.1";
    private int port = 5005;

    private Rigidbody rb;
    private float moveSpeed = 5f;
    private GameObject player;

    [Serializable]
    private class State
    {
        public float bot_x, bot_y, bot_z;
        public float player_x, player_y, player_z;
    }

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        player = GameObject.FindGameObjectWithTag("Player");  // Ensure the player has the tag "Player"

        try
        {
            client = new TcpClient(serverIP, port);
            stream = client.GetStream();
            Debug.Log("[DEBUG] Connected to Python server on " + serverIP + ":" + port);
        }
        catch (Exception e)
        {
            Debug.LogError("[ERROR] Could not connect to Python server: " + e.Message);
        }
    }

    void Update()
    {
        SendDataToPython();
        ReceiveActionFromPython();
    }

    private void SendDataToPython()
    {
        if (client == null || !client.Connected)
        {
            Debug.LogError("[ERROR] Python server is not connected!");
            return;
        }

        if (player == null)
        {
            Debug.LogError("[ERROR] Player object not found!");
            return;
        }

        State state = new State
        {
            bot_x = transform.position.x,
            bot_y = transform.position.y,
            bot_z = transform.position.z,
            player_x = player.transform.position.x,
            player_y = player.transform.position.y,
            player_z = player.transform.position.z
        };

        string json = JsonUtility.ToJson(state);
        byte[] data = Encoding.UTF8.GetBytes(json + "\n");

        try
        {
            stream.Write(data, 0, data.Length);
            Debug.Log("[DEBUG] Data sent to Python: " + json);
        }
        catch (Exception e)
        {
            Debug.LogError("[ERROR] Failed to send data: " + e.Message);
        }
    }

    private void ReceiveActionFromPython()
    {
        if (stream == null || !stream.DataAvailable) return;

        byte[] buffer = new byte[1024];
        int bytesRead = stream.Read(buffer, 0, buffer.Length);
        string action = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
        Debug.Log("[DEBUG] Received Action from Python: " + action);

        // Move bot based on received action
        Vector3 moveDirection = Vector3.zero;
        switch (action)
        {
            case "left": moveDirection = Vector3.left; break;
            case "right": moveDirection = Vector3.right; break;
            case "forward": moveDirection = Vector3.forward; break;
            case "backward": moveDirection = Vector3.back; break;
            case "fire": FireBullet(); break;
        }

        rb.linearVelocity = moveDirection * moveSpeed;
    }

    private void FireBullet()
    {
        Debug.Log("[DEBUG] AI Fired Bullet!");
        // Implement bullet firing logic (e.g., instantiate a bullet prefab)
    }

    void OnApplicationQuit()
    {
        if (client != null)
        {
            client.Close();
        }
    }
}
