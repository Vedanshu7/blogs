# The SQL Mistakes That Only Burn You in Production

*Six subtle bugs that pass code review, survive testing, and detonate at 3am.*

---

You've been writing SQL for years. You know about indexes. You know `NULL != NULL`. You've read the docs. And yet — there's a class of SQL bugs that are almost designed to hide from you. They work perfectly in development, pass QA, and sit dormant until your user base is large enough, or your data dirty enough, or your concurrency high enough to expose them.

This isn't a list of beginner mistakes. These are the ones that senior engineers ship.

---

## 1. Pagination Without a Stable Sort — Your Pages Are Drifting

Here's a query you've probably written a hundred times:

```sql
SELECT id, title, created_at
FROM posts
ORDER BY created_at DESC
LIMIT 20 OFFSET 40;
```

Looks fine. It works in development. In production, users report seeing the same post twice, or posts vanishing entirely between pages. You can't reproduce it.

Here's what's happening. When two rows share the same `created_at` value — and in a high-traffic app, they will — PostgreSQL has no rule for which one comes first. It picks based on whatever is cheapest at plan time, and that choice can change between two identical queries if statistics drift, a new index is created, or autovacuum runs. So the post that was 20th in your first query might be 21st in your second. Page 2 starts one row back, you see a duplicate. Or page 2 starts one row forward, a post disappears.

The fix is to always include a unique column as the final tiebreaker:

```sql
-- Still broken: OFFSET is O(N) and NULLs in created_at sort unpredictably
SELECT id, title, created_at
FROM posts
ORDER BY created_at DESC NULLS LAST, id DESC
LIMIT 20 OFFSET 40;
```

But there's a deeper problem: `OFFSET` itself is a lie at scale. `OFFSET 40000` tells Postgres to fetch 40,020 rows and throw away the first 40,000. Your "page 2000" query is doing a full scan. The right answer is keyset pagination:

```sql
-- Fast, stable, correct
SELECT id, title, created_at
FROM posts
WHERE (created_at, id) < ($last_seen_ts, $last_seen_id)
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

Pass the last row's `(created_at, id)` as the cursor. No duplicates, no skips, O(log N) with the right index. The cost: you can't jump to "page 47." But think about how often your users actually do that.

> **Pro tip:** Add `NULLS LAST` to every `ORDER BY` on a nullable column. Without it, NULLs sort to the top in `DESC` and to the bottom in `ASC` — almost never what you want, and always surprising.

---

## 2. `NOT IN` With a NULL in the Subquery — The Silent Zero-Row Bug

This one is a trap so elegant it almost deserves respect. Consider a feature-flag system:

```sql
SELECT id, email
FROM users
WHERE id NOT IN (SELECT user_id FROM blocked_users);
```

In your dev database, `blocked_users` has five rows, all with valid `user_id` values. The query returns the correct unblocked users. You ship it.

Three months later, a data import job inserts a row into `blocked_users` with a `NULL` `user_id` — maybe a user who deleted their account, maybe a migration gone slightly wrong. From that moment on, your query returns **zero rows**. Every user looks blocked. Your feature silently breaks for everyone.

The reason is SQL's three-valued logic. `NOT IN` expands to a series of `<>` comparisons. `5 <> NULL` doesn't evaluate to `TRUE` — it evaluates to `UNKNOWN`. And `TRUE AND UNKNOWN` is `UNKNOWN`. So no row can satisfy `NOT IN (... NULL ...)`. The optimizer doesn't warn you. The query doesn't error. It just returns nothing.

The fix is `NOT EXISTS`, which handles NULLs the way your intuition expects:

```sql
-- Safe: NOT EXISTS short-circuits correctly even with NULLs
SELECT u.id, u.email
FROM users u
WHERE NOT EXISTS (
  SELECT 1
  FROM blocked_users b
  WHERE b.user_id = u.id
);
```

Or, if you're attached to `NOT IN` for readability, filter the NULLs out of the subquery explicitly:

```sql
WHERE id NOT IN (
  SELECT user_id FROM blocked_users WHERE user_id IS NOT NULL
)
```

Both work. `NOT EXISTS` is generally faster on large datasets because Postgres can short-circuit on the first match rather than materializing the full set.

> **Pro tip:** Treat `NOT IN (subquery)` as a code smell. If there's any chance the subquery returns a nullable column — which is almost always — prefer `NOT EXISTS`.

---

## 3. Window Function Frame Defaults Are Not What You Think

Window functions feel magical the first time you use them. Then one day your running totals are wrong in a way that's hard to describe: they look almost right, but some values jump by too much.

Here's the query:

```sql
SELECT
  transaction_id,
  amount,
  created_at,
  SUM(amount) OVER (ORDER BY created_at) AS running_total
FROM transactions;
```

This looks like a running total. For most rows, it is. But watch what happens when two transactions happen at the exact same timestamp:

| transaction_id | amount | created_at          | running_total |
|----------------|--------|---------------------|---------------|
| 1              | 100    | 2024-01-01 10:00:00 | 100           |
| 2              | 200    | 2024-01-01 10:00:01 | 300           |
| 3              | 150    | 2024-01-01 10:00:02 | 675           |
| 4              | 225    | 2024-01-01 10:00:02 | 675           |

Rows 3 and 4 have the same timestamp. Both show `675`, which is the sum of all four rows. This is correct SQL behavior — but it's almost certainly not what you wanted.

The default window frame when you specify `ORDER BY` is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. With `RANGE`, "current row" means all rows that are peers — that is, all rows with the same `ORDER BY` value. Both timestamp-tied rows are peers, so both get the full sum through that point.

The fix is to use `ROWS` instead of `RANGE`:

```sql
SELECT
  transaction_id,
  amount,
  created_at,
  SUM(amount) OVER (
    ORDER BY created_at
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS running_total
FROM transactions;
```

`ROWS` treats each physical row as its own unit regardless of ties. Now row 3 gets the sum of rows 1–3, and row 4 gets the sum of rows 1–4. Which one is "right" depends on your use case — but at least it's deterministic.

> **Pro tip:** If you're using window aggregates on any column that isn't guaranteed unique across rows, always specify your frame explicitly. The default `RANGE` behavior is surprising enough that it should never be left implicit.

---

## 4. `UPDATE … FROM` With a Non-Unique Join — Arbitrary Data Corruption

This one is subtle enough that it's in the PostgreSQL documentation with a gentle warning that most people skim past.

Say you're building a bulk status-update system. A staging table comes in with new statuses for orders:

```sql
UPDATE orders
SET status = u.new_status
FROM order_updates u
WHERE orders.id = u.order_id;
```

Clean, readable, fast. You test it with your handcrafted test data where each `order_id` appears exactly once in `order_updates`. It works perfectly.

Then the ETL job that populates `order_updates` has a bug one night and inserts the same `order_id` twice — with two different `new_status` values. Your `UPDATE` runs. Postgres matches the order row against both staging rows and picks one. The docs say: *"which source row is used is unpredictable."* Not "the first one." Not "the last one." Whichever the query planner happens to visit first.

No error. No warning. Orders get assigned statuses at random. You find out when customers complain.

The safe pattern is to deduplicate the source before joining:

```sql
WITH deduped AS (
  SELECT DISTINCT ON (order_id) order_id, new_status
  FROM order_updates
  ORDER BY order_id, updated_at DESC  -- keep the most recent
)
UPDATE orders
SET status = d.new_status
FROM deduped d
WHERE orders.id = d.order_id;
```

`DISTINCT ON (order_id)` gives you exactly one row per order, and `ORDER BY updated_at DESC` lets you control which one wins. Now duplicates in the source are handled explicitly, not arbitrarily.

> **Pro tip:** Whenever you write `UPDATE … FROM`, immediately ask yourself: "Can this source table ever have more than one row per joined key?" If yes — and it usually can — add deduplication before the join.

---

## 5. Index Sabotage — Your Query Is Slower Than You Think

You added an index. The query is still slow. You run `EXPLAIN` and see `Seq Scan` where you expected `Index Scan`. What went wrong?

The most common culprit: you wrapped the indexed column in a function, or let the ORM do an implicit type cast.

```sql
-- Index on users(email) is completely ignored
SELECT * FROM users WHERE lower(email) = lower($1);

-- Index on events(created_at) is ignored
SELECT * FROM events WHERE created_at::date = '2024-01-15';

-- Index on products(id) may be ignored if $1 is bound as text
SELECT * FROM products WHERE id = $1;
```

PostgreSQL indexes are built on the stored values of columns. When you apply a function to a column in a `WHERE` clause, the planner can't use the existing index because the index doesn't store `lower(email)` — it stores `email`. It has to evaluate `lower(email)` for every row, which means a full sequential scan.

The fix depends on the case. For function-based lookups, build an expression index:

```sql
CREATE INDEX ON users (lower(email));

-- Now this query uses the index
SELECT * FROM users WHERE lower(email) = lower($1);
```

For the date truncation case, rewrite the predicate to use a range instead of a cast:

```sql
-- Uses the index on created_at
SELECT * FROM events
WHERE created_at >= '2024-01-15'
  AND created_at < '2024-01-16';
```

For type mismatch, fix the bind parameter type on the application side so no cast is needed.

The rule of thumb: never put the indexed column inside a function in the `WHERE` clause. Move the transformation to the other side of the comparison, or build an index that matches your transformation.

> **Pro tip:** Run `EXPLAIN (ANALYZE, BUFFERS)` — not just `EXPLAIN` — on your slow queries. The `BUFFERS` output shows how many disk pages were read. A `Seq Scan` that reads 50,000 buffers is a very different problem than one that reads 50.

---

## 6. Transaction Isolation and the Check-Then-Act Trap

Your application does this: check if something is available, then act on it. Book a seat, reserve inventory, claim a username. It works fine in testing. Under load, it occasionally does the wrong thing, and you can never reproduce the bug locally.

This is a classic race condition in the default `READ COMMITTED` isolation level.

```sql
-- Thread A and Thread B both run this at the same time
BEGIN;

SELECT seats_remaining FROM events WHERE id = 42;
-- Both see: 1

-- Both decide: "there's a seat available, I'll book it"

INSERT INTO bookings (event_id, user_id) VALUES (42, $user_id);
UPDATE events SET seats_remaining = seats_remaining - 1 WHERE id = 42;

COMMIT;
-- seats_remaining is now -1. Two people booked the last seat.
```

In `READ COMMITTED`, each statement sees the latest committed data at the time it executes. By the time Thread B runs its `UPDATE`, Thread A has already committed and decremented the count — but Thread B's `SELECT` had already decided to proceed. The check and the act are not atomic.

The fix is to lock the row at read time with `SELECT FOR UPDATE`:

```sql
BEGIN;

-- This acquires a row-level lock. Thread B blocks here until Thread A commits.
SELECT seats_remaining FROM events WHERE id = 42 FOR UPDATE;

-- Now only one thread at a time gets here
IF seats_remaining > 0 THEN
  INSERT INTO bookings (event_id, user_id) VALUES (42, $user_id);
  UPDATE events SET seats_remaining = seats_remaining - 1 WHERE id = 42;
END IF;

COMMIT;
```

The `FOR UPDATE` lock is released at commit. Thread B resumes, re-reads the (now-decremented) value, and correctly sees 0 seats. No double-booking.

For complex check-then-act patterns that span multiple tables, consider `REPEATABLE READ` or `SERIALIZABLE` isolation. `SERIALIZABLE` in PostgreSQL is genuinely serializable — it uses Serializable Snapshot Isolation and will automatically detect and abort one of two conflicting transactions. The trade-off: you need retry logic in your application.

> **Pro tip:** If your application ever does `SELECT` followed by `INSERT` or `UPDATE` based on that result, and both run inside the same transaction, you almost certainly need `SELECT FOR UPDATE` or a higher isolation level. "It works fine in testing" is not evidence it's safe under concurrency.

---

## Closing Thoughts

SQL is dangerously readable. You can look at a query and understand what it *should* do in about ten seconds. The gap between "what it should do" and "what it actually does under specific data conditions or concurrent load" is where these bugs live.

None of these are obscure edge cases invented for a gotcha article. Pagination drift happens at any scale. The `NOT IN` / NULL trap is one bad import away. Window frame defaults bite engineers who've been using window functions for years. Bulk update corruption is a standard ETL pattern. Index sabotage is invisible until your table has a million rows. And race conditions only show up when you have real concurrent users.

The common thread: these all behave correctly in development and fail in production. The only defense is understanding the engine well enough to know when your intuition is lying to you — and running `EXPLAIN ANALYZE` before you're sure you need it, not after.

---

*If any of these have burned you before, I'd love to hear which one in the comments. My personal nemesis was the window function frame — I had a finance dashboard running in production for three weeks before a sharp-eyed accountant noticed the numbers were off during peak hours.*
