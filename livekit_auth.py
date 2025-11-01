from livekit import api

def create_token(
    identity: str,
    room_name: str
) -> str:
    """LiveKit 토큰을 발급합니다.

    Args:
        identity (str): 참가자의 고유 식별자
        room_name (str): 입장할 방 이름

    Returns:
        str: 발급된 JWT 토큰
    """
    return api.AccessToken("devkey", "secret") \
        .with_identity(identity) \
        .with_grants(api.VideoGrants(
            room_create=True,
            room_join=True,
            room=room_name,
            can_publish=False,
            can_subscribe=True,
            can_publish_data=True
        )).to_jwt()