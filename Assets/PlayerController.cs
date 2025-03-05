using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [Header("Camera Settings")]
    public float NormalSpeed = 20f;
    public float BoostSpeed = 40f;
    public float SprintSpeed = 60f;
    public float LookSpeed = 2f;

    private float yaw = 0f;
    private float pitch = 0f;
    private float currentSpeed;

    void Start()
    {
        currentSpeed = NormalSpeed;
    }

    void Update()
    {
        yaw += Input.GetAxis("Mouse X") * LookSpeed;
        pitch -= Input.GetAxis("Mouse Y") * LookSpeed;
        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);

        currentSpeed = NormalSpeed;

        if (Input.GetKey(KeyCode.W))
        {
            currentSpeed = BoostSpeed;
            if (Input.GetKey(KeyCode.LeftShift))
            {
                currentSpeed = SprintSpeed;
            }
        }

        Vector3 moveDirection = transform.forward;

        if (Input.GetKey(KeyCode.S)) moveDirection -= transform.forward;
        if (Input.GetKey(KeyCode.A)) moveDirection -= transform.right;
        if (Input.GetKey(KeyCode.D)) moveDirection += transform.right;
        if (Input.GetKey(KeyCode.E)) moveDirection += transform.up;
        if (Input.GetKey(KeyCode.Q)) moveDirection -= transform.up;

        moveDirection.Normalize();
        transform.position += moveDirection * (currentSpeed * Time.deltaTime);
    }
}
