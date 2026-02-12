import os
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

RAW_PATH = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed")

IDLE_THRESHOLD = 2.0      # seconds
LARGE_PKT = 1000          # bytes
SMALL_PKT = 200           # bytes


def classify_event(row, prev_time):
    """Classify packet into event type"""
    if prev_time is not None and (row["timestamp"] - prev_time) > IDLE_THRESHOLD:
        return "IDLE"

    if row["protocol"] == 17:
        return "UDP_PACKET"

    if row["flags"] == "S":
        return "TCP_SYN"
    if row["flags"] == "SA":
        return "TCP_HANDSHAKE"
    if row["flags"] in ("F", "FA", "R", "RA"):
        return "SESSION_END"

    if row["length"] > LARGE_PKT:
        return "LARGE_TRANSFER"
    if row["length"] < SMALL_PKT:
        return "SMALL_TRANSFER"

    return "PACKET"


def parse_pcap(pcap_file):
    """Parse pcap file and extract ALL relevant features"""
    packets = rdpcap(pcap_file)
    rows = []

    for pkt in tqdm(packets, desc=f"Parsing {os.path.basename(pcap_file)}"):
        if IP not in pkt:
            continue

        # Base IP layer info
        ip_layer = pkt[IP]
        
        row = {
            # Timestamp
            "timestamp": float(pkt.time),
            
            # IP addresses
            "src_ip": ip_layer.src,
            "dst_ip": ip_layer.dst,
            
            # Protocol
            "protocol": ip_layer.proto,
            
            # Packet size info - CRITICAL for bytes_transferred
            "length": len(pkt),           # Total packet length
            "payload_length": len(pkt) - len(ip_layer),  # Payload size
            "ip_header_length": ip_layer.ihl * 4,  # IP header length
            
            # Defaults
            "src_port": None,
            "dst_port": None,
            "flags": None,
            "tcp_window": None,
            "tcp_options": None,
            "udp_length": None
        }

        # TCP specific info
        if TCP in pkt:
            tcp = pkt[TCP]
            row["src_port"] = tcp.sport
            row["dst_port"] = tcp.dport
            row["flags"] = str(tcp.flags)
            row["tcp_window"] = tcp.window
            row["tcp_options"] = str(tcp.options) if tcp.options else None
            
        # UDP specific info
        elif UDP in pkt:
            udp = pkt[UDP]
            row["src_port"] = udp.sport
            row["dst_port"] = udp.dport
            row["udp_length"] = udp.len

        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Add event classification
    df = classify_events(df)
    
    return df


def classify_events(df):
    """Add event type column to dataframe"""
    df = df.sort_values("timestamp")
    
    events = []
    prev_time = None
    
    for _, row in df.iterrows():
        evt = classify_event(row, prev_time)
        events.append(evt)
        prev_time = row["timestamp"]
    
    df["event"] = events
    return df


def build_sessions(df, source_name):
    """Build sessions with ALL features needed for RCA and clustering"""
    
    # Create session ID
    df["session_id"] = (
        df["src_ip"] + "_" +
        df["dst_ip"] + "_" +
        df["src_port"].astype(str) + "_" +
        df["dst_port"].astype(str) + "_" +
        df["protocol"].astype(str)
    )

    session_rows = []
    
    # Process each session
    for session_id, group in df.groupby("session_id"):
        group = group.sort_values("timestamp")
        
        # Session-level stats
        session_start = group["timestamp"].iloc[0]
        session_end = group["timestamp"].iloc[-1]
        duration = session_end - session_start
        
        # Calculate bytes transferred (sum of packet lengths)
        bytes_sent = group["length"].sum()
        avg_packet_size = group["length"].mean()
        
        # Calculate packet rates
        if duration > 0:
            packet_rate = len(group) / duration
            bytes_rate = bytes_sent / duration
        else:
            packet_rate = 0
            bytes_rate = 0
        
        # Get unique destinations
        unique_dst_ips = group["dst_ip"].nunique()
        unique_dst_ports = group["dst_port"].nunique()
        
        # Calculate burstiness (max packets in 1-second window)
        if len(group) > 1:
            timestamps = group["timestamp"].values
            max_packets_per_sec = 0
            for t in timestamps:
                count_in_window = ((timestamps >= t) & (timestamps < t + 1)).sum()
                max_packets_per_sec = max(max_packets_per_sec, count_in_window)
        else:
            max_packets_per_sec = 1
        
        # Process each packet in session
        prev_time = None
        for _, row in group.iterrows():
            evt = row["event"]
            
            session_rows.append({
                # Session identifiers
                "session_id": session_id,
                "source_pcap": source_name,
                
                # Event info
                "timestamp": row["timestamp"],
                "event": evt,
                
                # Packet features
                "length": row["length"],
                "payload_length": row["payload_length"],
                
                # Network context
                "src_ip": row["src_ip"],
                "dst_ip": row["dst_ip"],
                "src_port": row["src_port"],
                "dst_port": row["dst_port"],
                "protocol": row["protocol"],
                
                # TCP specific (if available)
                "tcp_flags": row["flags"],
                "tcp_window": row["tcp_window"],
                
                # Session-level aggregates (repeated for each row for easier analysis)
                "session_start": session_start,
                "session_end": session_end,
                "session_duration": duration,
                "session_bytes_total": bytes_sent,
                "session_packets_total": len(group),
                "session_avg_packet_size": avg_packet_size,
                "session_packet_rate": packet_rate,
                "session_bytes_rate": bytes_rate,
                "session_unique_dst_ips": unique_dst_ips,
                "session_unique_dst_ports": unique_dst_ports,
                "session_max_packets_per_sec": max_packets_per_sec
            })

    return pd.DataFrame(session_rows)


if __name__ == "__main__":
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    all_sessions = []
    
    for pcap in os.listdir(RAW_PATH):
        if not pcap.endswith(".pcap"):
            continue

        print(f"[INFO] Processing {pcap}")
        
        # Parse packets
        df_packets = parse_pcap(os.path.join(RAW_PATH, pcap))
        
        # Build sessions with ALL features
        df_sessions = build_sessions(df_packets, source_name=pcap)
        
        all_sessions.append(df_sessions)

    # Merge everything
    final_df = pd.concat(all_sessions, ignore_index=True)

    # Save detailed sessions
    out_file = os.path.join(PROCESSED_PATH, "all_sessions_detailed.csv")
    final_df.to_csv(out_file, index=False)
    
    # Also save a lightweight version for backward compatibility
    light_df = final_df[["session_id", "timestamp", "event", "length", "source_pcap"]].copy()
    light_file = os.path.join(PROCESSED_PATH, "all_sessions.csv")
    light_df.to_csv(light_file, index=False)

    print(f"\n[✅ DONE] Saved detailed sessions:")
    print(f"     {out_file}")
    print(f"     Total events: {len(final_df)}")
    print(f"     Total sessions: {final_df['session_id'].nunique()}")
    print(f"     Features per row: {len(final_df.columns)}")