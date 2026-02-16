import argparse
import socks
import socket
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_proxy(proxy):
    try:
        # Remove 'socks5://' from the proxy string
        proxy = proxy.replace('socks5://', '')
        # Split the proxy into host and port
        host, port = proxy.split(':')
        port = int(port)

        # Create a socket to connect through the proxy
        socks.set_default_proxy(socks.SOCKS5, host, port)
        socket.socket = socks.socksocket

        # Measure the connection time
        start_time = time.time()
        sock = socket.socket()
        sock.settimeout(5)  # seconds
        sock.connect(("cows.info.gf", 90))
        sock.close()
        end_time = time.time()

        # Calculate response time in milliseconds
        response_time = (end_time - start_time) * 1000
        return f"{proxy} - {int(response_time)}ms"
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description='SOCKS Proxy Checker')
    parser.add_argument('-l', '--list', required=True, help='File containing list of proxies')
    parser.add_argument('--threads', type=int, default=10, help='Number of threads to use for checking proxies')
    args = parser.parse_args()

    # Read the list of proxies from the file
    with open(args.list, 'r') as file:
        proxies = file.read().splitlines()

    results = []
    
    # Use ThreadPoolExecutor to manage multithreading
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(check_proxy, proxy): proxy for proxy in proxies}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking proxies", unit="proxy"):
            result = future.result()
            if result:
                results.append(result)

    # Print all the working proxies with their response times
    if results:
        print("\nWorking proxies:")
        for result in results:
            print(result)
    else:
        print("\nNo working proxies found.")

if __name__ == '__main__':
    main()
