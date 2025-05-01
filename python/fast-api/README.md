# Fast API

```bash
python3
from app.db.session import Engine
from app.db.models import Base
Base.metadata.create_all(bind=Engine)
quit()
```
