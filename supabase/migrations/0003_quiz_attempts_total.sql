-- supabase/migrations/0003_quiz_attempts_total.sql
-- Add the number of questions per attempt so quiz progress can be reported
-- as a percentage (score / total), not just a raw correct-count.
alter table quiz_attempts add column total int;
