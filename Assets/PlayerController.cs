using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [Header("Movement Settings")]
    public float baseSpeed = 10f;
    public float speedAdjustStep = 2f;
    public float boostMultiplier = 2f;
    public float strafeSpeed = 5f;

    [Header("Mouse Look Settings")]
    public float mouseSensitivity = 2f;

    [Header("Shooting Settings")]
    public GameObject bulletPrefab;
    public Transform firePoint;

    private float currentSpeed;
    private Rigidbody rb;

    private float yaw = 0f;
    private float pitch = 0f;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            Debug.LogError("Rigidbody component missing from PlayerController.");
        }

        Cursor.lockState = CursorLockMode.Locked;
        currentSpeed = baseSpeed;
    }

    void Update()
    {
        HandleMouseLook();
        HandleSpeedAdjustment();
        HandleShooting();
    }

    void FixedUpdate()
    {
        HandleMovement();
    }

    void HandleMouseLook()
    {
        yaw += Input.GetAxis("Mouse X") * mouseSensitivity;
        pitch -= Input.GetAxis("Mouse Y") * mouseSensitivity;
        pitch = Mathf.Clamp(pitch, -89f, 89f); // Prevent flipping

        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
    }

    void HandleSpeedAdjustment()
    {
        if (Input.GetKeyDown(KeyCode.W))
        {
            currentSpeed += speedAdjustStep;
        }

        if (Input.GetKeyDown(KeyCode.S))
        {
            currentSpeed = 0f;
            rb.linearVelocity = Vector3.zero;  // <--- Force stop when S is pressed
        }
    }

    void HandleMovement()
    {
        float strafe = 0f;
        if (Input.GetKey(KeyCode.A)) strafe = -strafeSpeed;
        if (Input.GetKey(KeyCode.D)) strafe = strafeSpeed;

        float speed = currentSpeed;
        if (Input.GetKey(KeyCode.LeftShift)) speed *= boostMultiplier;

        Vector3 forwardMovement = transform.forward * speed;
        Vector3 strafeMovement = transform.right * strafe;

        Vector3 finalVelocity = forwardMovement + strafeMovement;

        rb.linearVelocity = finalVelocity;
    }

    void HandleShooting()
    {
        if (Input.GetMouseButtonDown(0)) // Left click
        {
            Fire();
        }
    }

    void Fire()
    {
        if (bulletPrefab != null && firePoint != null)
        {
            Instantiate(bulletPrefab, firePoint.position, firePoint.rotation);
        }
    }

    void OnApplicationQuit()
    {
        Cursor.lockState = CursorLockMode.None;
    }
}
