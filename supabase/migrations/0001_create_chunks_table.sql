-- supabase/migrations/0001_create_chunks_table.sql
create extension if not exists vector;

create table chunks (
  id bigint generated always as identity primary key,
  chunk_id text not null unique,
  board text not null,
  class text not null,
  subject text not null,
  book_title text not null,
  chapter_number int not null,
  chapter_name text not null,
  topic text,
  chunk_type text not null,
  page_start int not null,
  page_end int not null,
  chunk_text text not null,
  embedding vector(768) not null
);

create index on chunks using hnsw (embedding vector_cosine_ops);
