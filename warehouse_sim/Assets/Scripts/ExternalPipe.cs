using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using UnityEngine;

public class ExternalPipe : MonoBehaviour
{
    public string remoteAddress = "127.0.0.1";
    public int remotePort = 12345;
    public CarController car;

    private Socket socket;
    private bool wasClosed = false;

    // A circular buffer of bytes coming from the other end
    private const int INCOMING_MESSAGE_SIZE = 8;
    private byte[] incomingMessage = new byte[INCOMING_MESSAGE_SIZE];
    private int incomingOffset = 0;

    void Start()
    {
        return;
        socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        socket.NoDelay = true; // Is this a good idea?
        socket.Connect(remoteAddress, remotePort);
    }

    void Update()
    {
        return;
        if (wasClosed)
        {
            return;
        }

        int pendingBytes = socket.Available;
        if (pendingBytes == 0)
        {
            return;
        }

        byte[] bytes = new byte[pendingBytes];
        int count = 0;

        try
        {
            count = socket.Receive(bytes);
        }
        catch (SocketException)
        {
            Debug.Log("ExternalPipe: Receive threw SocketException");
            wasClosed = true;
            return;
        }

        if (count == 0)
        {
            Debug.Log("ExternalPipe: Disconnected");
            wasClosed = true;
            return;
        }

        for (int i = 0; i < count; i++)
        {
            incomingMessage[incomingOffset++] = bytes[i];

            // Tell the handler when we have a complete message
            if (incomingOffset == INCOMING_MESSAGE_SIZE)
            {
                if (car != null)
                {
                    car.HandleExternalMessage(incomingMessage);
                }

                // Wrap back around for the next message
                incomingOffset = 0;
            }
        }
    }

    // Returns false if sending failed
    public bool Send(byte[] bytes)
    {
        if (wasClosed)
        {
            return false;
        }

        try
        {
            socket.Send(bytes);
        }
        catch (SocketException)
        {
            Debug.Log("ExternalPipe: Send threw SocketException");
            wasClosed = true;
            return false;
        }

        return true;
    }
}