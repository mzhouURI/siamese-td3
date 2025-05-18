import rosbag2_py
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_controller_values(bag_path, ns, c_topic, m_topic, e_topic):
    c_topic = ns + c_topic
    m_topic = ns + m_topic
    e_topic = ns + e_topic
    
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format="")

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    c_topic_type = None
    m_topic_type = None
    e_topic_type = None

    # Find the message type

    topics_to_find = {c_topic, m_topic, e_topic}

    for t in topics:
        if t.name == c_topic:
            c_topic_type = t.type
            topics_to_find.discard(c_topic)  # Remove from the set

        if t.name == m_topic:
            m_topic_type = t.type
            topics_to_find.discard(m_topic)

        if t.name == e_topic:
            e_topic_type = t.type
            topics_to_find.discard(e_topic)
    # Break early if all topics are found
        if not topics_to_find:
            break

    if not c_topic_type:
        print(f"Topic '{c_topic}' not found in bag.")
        return []
    
    if not m_topic_type:
        print(f"Topic '{m_topic}' not found in bag.")
        return []
    
    if not e_topic_type:
        print(f"Topic '{e_topic}' not found in bag.")
        return []
    
    #get message
    c_msg_type = get_message(c_topic_type)
    m_msg_type = get_message(m_topic_type)
    e_msg_type = get_message(e_topic_type)
    #read the topic
    c_values = []
    m_values = []
    e_values = []

    #get the start time
    if reader.has_next():
        topic_name, msg_data, t_ref = reader.read_next()
    # Convert the timestamp from nanoseconds to seconds


    while reader.has_next():
        topic_name, msg_data, t = reader.read_next()
        t= (t-t_ref)/1e9
        #set_point
        if topic_name == c_topic:
            msg = deserialize_message(msg_data, c_msg_type)
            x, y, z = (msg.position.x, 
                       msg.position.y, 
                       msg.position.z)
            roll, pitch, yaw = (msg.orientation.x,  
                                msg.orientation.y, 
                                msg.orientation.z)
            u,v,w = (msg.velocity.x, msg.velocity.y, msg.velocity.z)
            p,q,r = (msg.angular_rate.x, msg.angular_rate.y, msg.angular_rate.z)

            c_values.append((t, x, y, z, roll, pitch, yaw, u, v, w, p, q, r))
        
        if topic_name == m_topic:
            msg = deserialize_message(msg_data, m_msg_type)
            x, y, z = (msg.position.x, 
                       msg.position.y, 
                       msg.position.z)
            roll, pitch, yaw = (msg.orientation.x,  
                                msg.orientation.y, 
                                msg.orientation.z)
            u,v,w = (msg.velocity.x, msg.velocity.y, msg.velocity.z)
            p,q,r = (msg.angular_rate.x, msg.angular_rate.y, msg.angular_rate.z)

            m_values.append((t, x, y, z, roll, pitch, yaw, u, v, w, p, q, r))

        if topic_name == e_topic:
            msg = deserialize_message(msg_data, e_msg_type)
            x, y, z = (msg.position.x, 
                       msg.position.y, 
                       msg.position.z)
            roll, pitch, yaw = (msg.orientation.x,  
                                msg.orientation.y, 
                                msg.orientation.z)
            u,v,w = (msg.velocity.x, msg.velocity.y, msg.velocity.z)
            p,q,r = (msg.angular_rate.x, msg.angular_rate.y, msg.angular_rate.z)

            e_values.append((t, x, y, z, roll, pitch, yaw, u, v, w, p, q, r))
    

    c_values = np.array(c_values)
    m_values = np.array(m_values)
    e_values = np.array(e_values)
    data_mean = np.mean(e_values[:,3:7], axis=0)
    data_rms =np.sqrt(np.mean(np.square(e_values[:,3:7]), axis=0))
    print("Error in z, roll, pitch, yaw, surge")
    print("Control error Mean:", ", ".join(f"{d:.4f}" for d in data_mean))
    print("Control error RMS:", ", ".join(f"{d:.4f}" for d in data_rms))
    
    return c_values, m_values, e_values


#all float
def read_thrusters(bag_path, ns, topics):
    topics = [ns + entry for entry in topics]
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format="")

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    bag_topics = reader.get_all_topics_and_types()
    topic_type = None
    for t in bag_topics:
        if t.name in topics:
            topic_type = t.type
            topic_type = get_message(topic_type)
            # print(topic_type)
            break
    # print(topic_type)
    
    topics_in_bag = {topic.name for topic in bag_topics}  # Use a set for quick lookup     

    missing_topics = [topic for topic in topics if topic not in topics_in_bag]

    if missing_topics:
        print(f"Missing topics: {missing_topics}")
    else:
        print("All expected topics are in the bag.")


    #get the start time
    if reader.has_next():
        topic_name, msg_data, t_ref = reader.read_next()

    #initialize 2d list
    thruster_data = [[0.0] * len(topics) for _ in range(len(topics))] 
    thruster_data_t = [[0.0] * len(topics) for _ in range(len(topics))] 
    # print(thruster_data)
    # count = int(np.zeros( len(topics) ))
    counter = [0]*len(topics)
  
    while reader.has_next():
        topic_name, msg_data, t = reader.read_next()
        t = (t-t_ref)/1e9

        if topic_name in topics:
            msg = deserialize_message(msg_data, topic_type)

            i = topics.index(topic_name)  # Get the index

            if len(thruster_data)<=counter[i]:
                thruster_data.append([0]*len(topics))
                thruster_data_t.append([0]*len(topics))

            thruster_data[counter[i]][i] = msg.data
            thruster_data_t[counter[i]][i] = t 

            counter[i] += 1

    thruster_data =np.array(thruster_data)
    thruster_data_t = np.array(thruster_data_t)
    data_mean = np.mean(thruster_data, axis=0)
    data_rms =np.sqrt(np.mean(np.square(thruster_data), axis=0))
    print("Thruster  Mean:", ", ".join(f"{d:.4f}" for d in data_mean))
    print("Thruster RMS:", ", ".join(f"{d:.4f}" for d in data_rms))
    

    
    return thruster_data_t, thruster_data


def read_joints(bag_path, ns, topic):
    topic = ns+ topic
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format="")

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    topic_type = None

    # Find the message type


    for t in topics:
        if t.name == topic:
            topic_type = t.type
            break

    if not topic_type:
        print(f"Topic '{topic_type}' not found in bag.")
        return []
    
    #get message
    c_msg_type = get_message(topic_type)
    #read the topic
    c_values = []

    #get the start time
    if reader.has_next():
        topic_name, msg_data, t_ref = reader.read_next()
    # Convert the timestamp from nanoseconds to seconds


    while reader.has_next():
        topic_name, msg_data, t = reader.read_next()
        t= (t-t_ref)/1e9
        #set_point
        if topic_name == topic:
            msg = deserialize_message(msg_data, c_msg_type)
            s1 = msg.position[0]
            s2 = msg.position[1]
            
            s3 = msg.position[2]
            s4 = msg.position[3]
            c_values.append((t, s1, s2, s3, s4))

            # c_values.append((t, s1, s2))


    c_values = np.array(c_values)
    d_value = np.diff(c_values[:,1:]*180/np.pi, axis=0)
    data_mean = np.mean(d_value, axis=0)
    data_rms =np.sqrt(np.mean(np.square(d_value), axis=0))
    # print("Servo  angle delta mean:", ", ".join(f"{d:.6f}" for d in data_mean))
    # print("Servo  angle delta rms:", ", ".join(f"{d:.6f}" for d in data_rms))

    return c_values