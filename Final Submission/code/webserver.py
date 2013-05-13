import threading
import webbrowser
import BaseHTTPServer
import SimpleHTTPServer

from data_reader import *
from neural_net import *
from neural_net_impl import *

FILE = 'index.html'
PORT = 8080


class TestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    """The test example handler."""

    def do_POST(self):
        """Handle a post request by returning the square of the number."""
        length = int(self.headers.getheader('content-length'))        
        data_string = self.rfile.read(length)
        type_input = data_string[0]
        data_string = data_string[1:]
        data_string = data_string.split(',')
        for i in range(len(data_string)):
            data_string[i] = 255 - int(data_string[i])
        # Creating the data structure for our neural network on the server
        network = HiddenNetwork()
        network.FeedForwardFn = FeedForward
        if type_input == 'g':
            network.InitializeReadWeights('g_weights')
        else:
            network.InitializeReadWeights('b_weights')
        print ("webserver.py datastring:"), data_string
        digit_guess = network.Classify(data_string, "UI")
        print digit_guess
        try:
            result = int(digit_guess)
            #result = int(data_string) ** 2
        except:
            result = 'error'
        self.wfile.write(result)


def open_browser():
    """Start a browser after waiting for half a second."""
    def _open_browser():
        webbrowser.open('http://localhost:%s/%s' % (PORT, FILE))
    thread = threading.Timer(0.5, _open_browser)
    thread.start()

def start_server():
    """Start the server."""
    server_address = ("", PORT)
    server = BaseHTTPServer.HTTPServer(server_address, TestHandler)
    server.serve_forever()

if __name__ == "__main__":
    open_browser()
    start_server()