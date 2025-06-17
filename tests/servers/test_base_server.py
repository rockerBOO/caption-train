"""Tests for base server classes."""

import pytest
from caption_train.servers.base import CaptionServerApp, ImageGalleryServer, CaptionGenerationServer


def test_caption_server_app_state_management():
    """Test state management for server applications."""
    server = CaptionServerApp()
    
    # Test setting and getting state
    server.set_state('test_key', 'test_value')
    assert server.get_state('test_key') == 'test_value'
    
    # Test getting non-existent state with default
    assert server.get_state('non_existent', 'default') == 'default'
    
    # Test overwriting state
    server.set_state('test_key', 'new_value')
    assert server.get_state('test_key') == 'new_value'


def test_image_gallery_server_initialization():
    """Test ImageGalleryServer initialization."""
    server = ImageGalleryServer(images_dir='/test/images')
    
    assert server.get_state('images_dir') == '/test/images'


def test_caption_generation_server_initialization():
    """Test CaptionGenerationServer initialization."""
    server = CaptionGenerationServer(model_id='test-model')
    
    assert server.get_state('model_id') == 'test-model'


def test_server_route_registration_not_implemented():
    """Test that base server requires route registration."""
    with pytest.raises(NotImplementedError):
        server = CaptionServerApp()
        server.register_routes()