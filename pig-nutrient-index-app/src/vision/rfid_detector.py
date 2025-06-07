import time
import hashlib
from collections import defaultdict
import cv2

class RFIDDetector:
    def __init__(self, rfid_reader_port=None):
        """
        RFID detector for identifying individual pigs.
        Uses RFID tags for reliable identification in farm environments.
        
        Parameters:
        rfid_reader_port: Serial port for RFID reader (e.g., 'COM3' or '/dev/ttyUSB0')
        """
        self.rfid_reader_port = rfid_reader_port
        self.pig_database = {}
        self.last_detection_time = {}
        self.detection_timeout = 5.0  # seconds
        
        # Common RFID tag prefixes for pig identification
        self.pig_tag_prefixes = ['PIG', 'SWINE', 'HOG']
        
        # Simulated RFID data for demo (in real implementation, this would come from hardware)
        self.simulated_tags = {
            'PIG001': {'weight_history': [], 'breed': 'Yorkshire', 'birth_date': '2024-01-15'},
            'PIG002': {'weight_history': [], 'breed': 'Hampshire', 'birth_date': '2024-01-20'},
            'PIG003': {'weight_history': [], 'breed': 'Duroc', 'birth_date': '2024-02-01'},
            'PIG004': {'weight_history': [], 'breed': 'Landrace', 'birth_date': '2024-02-10'},
            'PIG005': {'weight_history': [], 'breed': 'Yorkshire', 'birth_date': '2024-02-15'},
        }
        
    def initialize_rfid_reader(self):
        """Initialize the RFID reader hardware"""
        if self.rfid_reader_port:
            try:
                # In real implementation, initialize serial connection to RFID reader
                # import serial
                # self.rfid_connection = serial.Serial(self.rfid_reader_port, 9600, timeout=1)
                print(f"RFID reader initialized on port {self.rfid_reader_port}")
                return True
            except Exception as e:
                print(f"Failed to initialize RFID reader: {e}")
                return False
        else:
            print("Using simulated RFID detection for demo")
            return True

    def read_rfid_tag(self, timeout=2.0):
        """
        Read RFID tag from hardware reader.
        
        Parameters:
        timeout: Maximum time to wait for tag detection
        
        Returns:
        tag_id: String ID of detected tag, or None if no tag detected
        """
        if self.rfid_reader_port:
            # Real implementation would read from serial port
            # In demo, we'll simulate detection
            return self._simulate_rfid_detection()
        else:
            # Simulation mode
            return self._simulate_rfid_detection()

    def _simulate_rfid_detection(self):
        """Simulate RFID tag detection for demo purposes"""
        import random
        
        # Simulate detection probability (80% chance of detecting a tag)
        if random.random() < 0.8:
            # Return a random pig tag
            return random.choice(list(self.simulated_tags.keys()))
        else:
            return None

    def detect_pig_in_camera_area(self, timeout=5.0):
        """
        Detect which pig is currently in the camera imaging area.
        
        Parameters:
        timeout: Maximum time to wait for detection
        
        Returns:
        pig_info: Dictionary with pig ID and metadata
        """
        print("Scanning for RFID tags in camera area...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            tag_id = self.read_rfid_tag(timeout=1.0)
            
            if tag_id and self._is_valid_pig_tag(tag_id):
                pig_info = {
                    'pig_id': tag_id,
                    'confidence': 1.0,  # RFID is highly reliable
                    'detection_method': 'rfid',
                    'detection_time': time.time(),
                    'metadata': self.simulated_tags.get(tag_id, {})
                }
                
                # Update detection history
                self.last_detection_time[tag_id] = time.time()
                
                print(f"✅ Pig detected: {tag_id}")
                return pig_info
            
            time.sleep(0.1)  # Brief pause between reads
        
        print("❌ No pig detected in camera area")
        return None

    def detect_pig_at_feeder(self, feeder_id, timeout=10.0):
        """
        Detect which pig is approaching a specific feeder.
        
        Parameters:
        feeder_id: ID of the feeder station
        timeout: Maximum time to wait for detection
        
        Returns:
        pig_info: Dictionary with pig ID and access permission
        """
        print(f"Monitoring feeder {feeder_id} for approaching pigs...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            tag_id = self.read_rfid_tag(timeout=1.0)
            
            if tag_id and self._is_valid_pig_tag(tag_id):
                # Check if this pig is authorized to eat
                feeding_status = self._check_feeding_authorization(tag_id)
                
                pig_info = {
                    'pig_id': tag_id,
                    'feeder_id': feeder_id,
                    'confidence': 1.0,
                    'detection_method': 'rfid',
                    'detection_time': time.time(),
                    'feeding_authorized': feeding_status['authorized'],
                    'reason': feeding_status['reason'],
                    'metadata': self.simulated_tags.get(tag_id, {})
                }
                
                print(f"🐷 Pig {tag_id} at feeder {feeder_id}")
                print(f"Feeding authorized: {feeding_status['authorized']} ({feeding_status['reason']})")
                
                return pig_info
            
            time.sleep(0.1)
        
        return None

    def _is_valid_pig_tag(self, tag_id):
        """Check if the tag ID belongs to a pig"""
        if not tag_id:
            return False
        
        # Check if tag starts with known pig prefixes
        for prefix in self.pig_tag_prefixes:
            if tag_id.startswith(prefix):
                return True
        
        # Check if it's in our simulated database
        return tag_id in self.simulated_tags

    def _check_feeding_authorization(self, pig_id):
        """
        Check if a pig is authorized to feed based on recent feeding history.
        
        Parameters:
        pig_id: ID of the pig
        
        Returns:
        dict: Authorization status and reason
        """
        current_time = time.time()
        
        # Check if pig was fed recently (within last 4 hours)
        if pig_id in self.last_detection_time:
            time_since_last_feed = current_time - self.last_detection_time[pig_id]
            
            if time_since_last_feed < 4 * 3600:  # 4 hours in seconds
                return {
                    'authorized': False,
                    'reason': f'Fed {time_since_last_feed/3600:.1f} hours ago. Wait {4 - time_since_last_feed/3600:.1f} hours.'
                }
        
        # Check daily feeding quota (example logic)
        # In real implementation, this would check database records
        return {
            'authorized': True,
            'reason': 'Feeding quota available'
        }

    def get_pig_metadata(self, pig_id):
        """Get stored metadata for a pig"""
        return self.simulated_tags.get(pig_id, {})

    def update_pig_record(self, pig_id, weight=None, feeding_time=None):
        """Update pig's records with new data"""
        if pig_id not in self.pig_database:
            self.pig_database[pig_id] = {
                'weight_history': [],
                'feeding_history': [],
                'first_seen': time.time()
            }
        
        if weight:
            self.pig_database[pig_id]['weight_history'].append({
                'weight': weight,
                'timestamp': time.time()
            })
        
        if feeding_time:
            self.pig_database[pig_id]['feeding_history'].append({
                'fed_time': feeding_time,
                'timestamp': time.time()
            })

    def visualize_detection(self, image, pig_info):
        """
        Draw RFID detection results on image for visualization.
        
        Parameters:
        image: Input image
        pig_info: Result from detect_pig_in_camera_area
        
        Returns:
        annotated_image: Image with detection results drawn
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        result_image = image.copy()
        
        if pig_info:
            # Draw RFID icon and information
            cv2.putText(result_image, f"RFID: {pig_info['pig_id']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw RFID symbol (simplified radio waves)
            center = (50, 80)
            for i in range(3):
                radius = 15 + i * 10
                cv2.circle(result_image, center, radius, (0, 255, 255), 2)
            
            # Draw confidence and method
            confidence_text = f"Confidence: {pig_info['confidence']:.2f} (RFID)"
            cv2.putText(result_image, confidence_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw metadata if available
            if 'metadata' in pig_info and pig_info['metadata']:
                metadata = pig_info['metadata']
                y_offset = 150
                
                if 'breed' in metadata:
                    cv2.putText(result_image, f"Breed: {metadata['breed']}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 25
                
                if 'birth_date' in metadata:
                    cv2.putText(result_image, f"Born: {metadata['birth_date']}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return result_image

    def get_system_status(self):
        """Get current system status and statistics"""
        return {
            'active_pigs': len(self.pig_database),
            'total_detections': sum(len(pig['weight_history']) for pig in self.pig_database.values()),
            'reader_connected': self.rfid_reader_port is not None,
            'known_tags': list(self.simulated_tags.keys())
        }

# Hardware Integration Examples (for real implementation)
class RealRFIDHardware:
    """Example integration with real RFID hardware"""
    
    @staticmethod
    def get_recommended_hardware():
        """Return recommended RFID hardware for pig farming"""
        return {
            'reader': {
                'model': 'ThingMagic M6e Nano',
                'frequency': '860-960 MHz (UHF)',
                'read_range': '1-6 meters',
                'interface': 'USB/Serial',
                'cost': '$200-400'
            },
            'tags': {
                'model': 'Smartrac DogBone RFID Tag',
                'type': 'Passive UHF',
                'mounting': 'Ear tag or collar',
                'durability': 'IP67 rated',
                'cost': '$2-5 per tag'
            },
            'antenna': {
                'model': 'Circular polarized panel antenna',
                'gain': '6-9 dBi',
                'coverage': '120° beam width',
                'mounting': 'Above camera and feeder areas'
            }
        }
    
    @staticmethod
    def get_installation_guide():
        """Return installation recommendations"""
        return {
            'camera_area': {
                'antenna_height': '2-3 meters above imaging area',
                'read_zone': '2x2 meter area for weight estimation',
                'power': '1-2 watts for reliable detection'
            },
            'feeder_area': {
                'antenna_placement': 'Near feeder entrance (1 meter)',
                'read_zone': '1 meter radius around feeder',
                'integration': 'Connect to feeder control system'
            },
            'networking': {
                'protocol': 'TCP/IP or Serial over RS485',
                'data_format': 'JSON with pig_id and timestamp',
                'redundancy': 'Multiple readers per area recommended'
            }
        } 