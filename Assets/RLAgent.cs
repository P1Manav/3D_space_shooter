using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RLAgent : Agent
{
    public Rigidbody rb;
    public GameObject bulletPrefab;
    public Transform bulletSpawn;
    private float moveSpeed = 10f;
    private float bulletSpeed = 20f;

    private GameObject player;

    public override void Initialize()
    {
        player = GameObject.FindGameObjectWithTag("Player");
        if (player == null)
        {
            Debug.LogError("[ERROR] Player object not found! Make sure your player has the tag 'Player'.");
        }

        // Start continuous firing
        StartCoroutine(FireContinuously());
    }

    public override void OnEpisodeBegin()
    {
        transform.position = new Vector3(Random.Range(-10f, 10f), Random.Range(-10f, 10f), Random.Range(-10f, 10f));
        rb.linearVelocity = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.position.x);
        sensor.AddObservation(transform.position.y);
        sensor.AddObservation(transform.position.z);

        if (player != null)
        {
            sensor.AddObservation(player.transform.position.x);
            sensor.AddObservation(player.transform.position.y);
            sensor.AddObservation(player.transform.position.z);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (player == null) return;

        Vector3 direction = (player.transform.position - transform.position).normalized;
        rb.linearVelocity = direction * moveSpeed;

        Debug.Log("[DEBUG] AI Position: " + transform.position);
        Debug.Log("[DEBUG] AI Velocity: " + rb.linearVelocity);
    }

    IEnumerator FireContinuously()
    {
        while (true)
        {
            FireBullet();
            yield return new WaitForSeconds(0.5f); // Fire every 0.5 seconds
        }
    }

    void FireBullet()
    {
        if (bulletPrefab == null || bulletSpawn == null)
        {
            Debug.LogError("[ERROR] BulletPrefab or BulletSpawn is not assigned.");
            return;
        }

        GameObject bullet = Instantiate(bulletPrefab, bulletSpawn.position, Quaternion.identity);
        bullet.GetComponent<Rigidbody>().linearVelocity = (player.transform.position - bulletSpawn.position).normalized * bulletSpeed;

        Debug.Log("[DEBUG] AI Fired Bullet!");
    }
}
