from enum import Enum

from marshmallow import Schema, fields

RequestFrom = Enum("from", ["client", "training"])
RequestType = Enum("type", ["caption_quality", "caption_new"])


class NewCaptionSchema(Schema):
    new_caption = fields.Str(required=True)


class CaptionQualitySchema(Schema):
    quality = fields.Int(required=True)


class CaptionQualityRequestSchema(Schema):
    req_from = fields.Constant(RequestFrom.client)
    type = fields.Constant(RequestType.caption_quality)
    payload = fields.Nested(CaptionQualitySchema)


class NewCaptionRequestSchema(Schema):
    req_from = fields.Constant(RequestFrom.client)
    type = fields.Constant(RequestType.caption_new)
    payload = fields.Nested(NewCaptionSchema)
