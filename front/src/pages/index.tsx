import { Center, Container, Button, Input } from "@chakra-ui/react";
import { useRouter } from "next/router";
import React, { useState, useContext } from "react";
import axios from "axios";
import { NameContext } from "../state/nameContext";

export default function Home() {
  const router = useRouter();
  const [file, setFile] = useState<File>();
  const { name, setName } = useContext(NameContext);

  const ok = async (e: any) => {
    e.preventDefault();
    setName(file?.name);
    const body = new FormData();
    body.append("file", file);
    await axios.post("http://127.0.0.1:5000/upload", body).catch().then();
    router.push("/result");
  };

  return (
    <Container>
      <Center bg="tomato" h="100px" color="white" fontSize="30">
        Portfolio
      </Center>
      <form onSubmit={ok}>
        <Input
          type="file"
          onChange={(e: any) => setFile(e.target.files[0])}
        ></Input>
        <Button type="submit">選択</Button>
      </form>
    </Container>
  );
}
