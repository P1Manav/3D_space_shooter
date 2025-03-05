using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform player;  // Reference to the player (spaceship)
    public Vector3 offset = new Vector3(0, 2, -5);  // Offset relative to player's orientation
    public float smoothSpeed = 5f;  // Camera follow smoothing

    void LateUpdate()
    {
        if (player == null) return;

        // Calculate offset relative to the player's orientation
        Vector3 relativeOffset = player.right * offset.x + player.up * offset.y + player.forward * offset.z;
        
        // Compute the target camera position
        Vector3 desiredPosition = player.position + relativeOffset;

        // Smoothly move the camera to the desired position
        transform.position = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed * Time.deltaTime);

        // Rotate the camera to always look at the player
        transform.LookAt(player.position);
    }
}
