-- supabase/migrations/0002_chat_quiz_tables_and_chunks_rls.sql

alter table chunks enable row level security;
create policy "public read access" on chunks
  for select to authenticated, anon
  using (true);

create table chat_history (
  id bigint generated always as identity primary key,
  user_id uuid not null,
  chapter_number int not null,
  subject text not null,
  class text not null,
  question text not null,
  answer text not null,
  created_at timestamptz not null default now()
);
alter table chat_history enable row level security;
create policy "select own chat history" on chat_history
  for select to authenticated
  using ((select auth.uid()) = user_id);
create policy "insert own chat history" on chat_history
  for insert to authenticated
  with check ((select auth.uid()) = user_id);

create table quiz_attempts (
  id bigint generated always as identity primary key,
  user_id uuid not null,
  chapter_number int not null,
  subject text not null,
  class text not null,
  quiz_json jsonb not null,
  student_answers jsonb,
  score int,
  attempted_at timestamptz not null default now()
);
alter table quiz_attempts enable row level security;
create policy "select own quiz attempts" on quiz_attempts
  for select to authenticated
  using ((select auth.uid()) = user_id);
create policy "insert own quiz attempts" on quiz_attempts
  for insert to authenticated
  with check ((select auth.uid()) = user_id);
create policy "update own quiz attempts" on quiz_attempts
  for update to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);
