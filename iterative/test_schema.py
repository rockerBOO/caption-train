from schema import CaptionQualityRequestSchema, NewCaptionRequestSchema


def test_caption_quality_schema():
    request = {
        "req_from": "client",
        "type": "caption.quality",
        "payload": {"quality": 1},
    }

    CaptionQualityRequestSchema().validate(request)


def test_new_caption_schema():
    request = {
        "req_from": "client",
        "type": "caption.new",
        "payload": {"new_caption": "a man sitting in a chair"},
    }

    NewCaptionRequestSchema().validate(request)
