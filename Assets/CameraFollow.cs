using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform target;          // Player spaceship
    public Vector3 offset = new Vector3(0f, 5f, -10f);  // Offset relative to the player
    public float followSpeed = 10f;   // How fast the camera follows the player
    public float rotationSpeed = 5f;  // Smooth rotation speed

    void LateUpdate()
    {
        if (target == null) return;

        // Calculate desired position based on player's rotation
        Vector3 desiredPosition = target.position + target.rotation * offset;
        transform.position = Vector3.Lerp(transform.position, desiredPosition, followSpeed * Time.deltaTime);

        // Rotate camera to look at the target
        Quaternion desiredRotation = Quaternion.LookRotation(target.position - transform.position);
        transform.rotation = Quaternion.Slerp(transform.rotation, desiredRotation, rotationSpeed * Time.deltaTime);
    }
}
